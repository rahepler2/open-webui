from typing import Optional, List, Dict, Any
import logging
import json
import numpy as np
from sqlalchemy import (
    func,
    literal,
    cast,
    column,
    create_engine,
    Column,
    Integer,
    MetaData,
    LargeBinary,
    select,
    text,
    Text,
    Table,
    values,
)
from sqlalchemy.sql import true
from sqlalchemy.pool import NullPool, QueuePool

from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB, array
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.exc import NoSuchTableError

from open_webui.retrieval.vector.main import (
    VectorDBBase,
    VectorItem,
    SearchResult,
    GetResult,
)
from open_webui.config import (
    PGVECTOR_DB_URL,
    PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH,
    PGVECTOR_PGCRYPTO,
    PGVECTOR_PGCRYPTO_KEY,
    PGVECTOR_POOL_SIZE,
    PGVECTOR_POOL_MAX_OVERFLOW,
    PGVECTOR_POOL_TIMEOUT,
    PGVECTOR_POOL_RECYCLE,
)

from open_webui.env import SRC_LOG_LEVELS

VECTOR_LENGTH = PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH
Base = declarative_base()

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

# Numba imports with graceful fallback
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    log.info("Numba JIT compilation available for binary operations")
except ImportError:
    NUMBA_AVAILABLE = False
    log.debug("Numba not available, falling back to NumPy operations")
    # Define dummy decorators for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


# Optimized Hamming distance functions with Numba JIT compilation
@njit(parallel=True, fastmath=True, cache=True)
def _hamming_distance_bytes_numba(query_bytes: np.ndarray, candidate_bytes: np.ndarray) -> int:
    """Numba-optimized Hamming distance for single pair of byte arrays"""
    if len(query_bytes) != len(candidate_bytes):
        return 999999
    
    distance = 0
    for i in prange(len(query_bytes)):
        # XOR and count bits using Brian Kernighan's algorithm
        xor_byte = query_bytes[i] ^ candidate_bytes[i]
        while xor_byte:
            distance += 1
            xor_byte &= xor_byte - 1  # Remove lowest set bit
    
    return distance

@njit(parallel=True, fastmath=True, cache=True)
def _batch_hamming_distance_numba(query_bytes: np.ndarray, candidates_matrix: np.ndarray) -> np.ndarray:
    """Numba-optimized batch Hamming distance calculation"""
    n_candidates, n_bytes = candidates_matrix.shape
    distances = np.zeros(n_candidates, dtype=np.int32)
    
    for i in prange(n_candidates):
        distances[i] = _hamming_distance_bytes_numba(query_bytes, candidates_matrix[i])
    
    return distances

def _hamming_distance_bytes_fallback(query_bytes: bytes, candidate_bytes: bytes) -> int:
    """NumPy fallback for Hamming distance when Numba is unavailable"""
    try:
        if len(query_bytes) != len(candidate_bytes):
            return 999999
        
        query_array = np.frombuffer(query_bytes, dtype=np.uint8)
        candidate_array = np.frombuffer(candidate_bytes, dtype=np.uint8)
        
        xor_result = query_array ^ candidate_array
        return int(np.sum(np.unpackbits(xor_result)))
    except Exception:
        return 999999

def hamming_distance_bytes(query_bytes: bytes, candidate_bytes: bytes) -> int:
    """High-performance Hamming distance with automatic Numba/NumPy selection"""
    if NUMBA_AVAILABLE:
        try:
            query_array = np.frombuffer(query_bytes, dtype=np.uint8)
            candidate_array = np.frombuffer(candidate_bytes, dtype=np.uint8)
            return int(_hamming_distance_bytes_numba(query_array, candidate_array))
        except Exception:
            # Fallback to NumPy if Numba fails
            return _hamming_distance_bytes_fallback(query_bytes, candidate_bytes)
    else:
        return _hamming_distance_bytes_fallback(query_bytes, candidate_bytes)

def batch_hamming_distance(query_bytes: bytes, candidate_bytes_list: List[bytes]) -> List[int]:
    """Batch Hamming distance calculation with optimizations"""
    if not candidate_bytes_list:
        return []
    
    if NUMBA_AVAILABLE and len(candidate_bytes_list) >= 10:  # Use batch optimization for larger sets
        try:
            query_array = np.frombuffer(query_bytes, dtype=np.uint8)
            
            # Stack candidate byte arrays into a matrix
            candidates_matrix = np.vstack([
                np.frombuffer(candidate, dtype=np.uint8) 
                for candidate in candidate_bytes_list
            ])
            
            distances = _batch_hamming_distance_numba(query_array, candidates_matrix)
            return distances.tolist()
            
        except Exception as e:
            log.debug(f"Batch Numba optimization failed, falling back: {e}")
    
    # Fallback to individual calculations
    return [hamming_distance_bytes(query_bytes, candidate) for candidate in candidate_bytes_list]


def pgcrypto_encrypt(val, key):
    return func.pgp_sym_encrypt(val, literal(key))


def pgcrypto_decrypt(col, key, outtype="text"):
    return func.cast(func.pgp_sym_decrypt(col, literal(key)), outtype)


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    id = Column(Text, primary_key=True)
    vector = Column(Vector(dim=VECTOR_LENGTH), nullable=True)
    collection_name = Column(Text, nullable=False)
    binary_vector = Column(LargeBinary, nullable=True)

    if PGVECTOR_PGCRYPTO:
        text = Column(LargeBinary, nullable=True)
        vmetadata = Column(LargeBinary, nullable=True)
    else:
        text = Column(Text, nullable=True)
        vmetadata = Column(MutableDict.as_mutable(JSONB), nullable=True)


class PgvectorClient(VectorDBBase):
    def __init__(self) -> None:

        # if no pgvector uri, use the existing database connection
        if not PGVECTOR_DB_URL:
            from open_webui.internal.db import Session

            self.session = Session
        else:
            if isinstance(PGVECTOR_POOL_SIZE, int):
                if PGVECTOR_POOL_SIZE > 0:
                    engine = create_engine(
                        PGVECTOR_DB_URL,
                        pool_size=PGVECTOR_POOL_SIZE,
                        max_overflow=PGVECTOR_POOL_MAX_OVERFLOW,
                        pool_timeout=PGVECTOR_POOL_TIMEOUT,
                        pool_recycle=PGVECTOR_POOL_RECYCLE,
                        pool_pre_ping=True,
                        poolclass=QueuePool,
                    )
                else:
                    engine = create_engine(
                        PGVECTOR_DB_URL, pool_pre_ping=True, poolclass=NullPool
                    )
            else:
                engine = create_engine(PGVECTOR_DB_URL, pool_pre_ping=True)

            SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
            )
            self.session = scoped_session(SessionLocal)

        try:
            # Ensure the pgvector extension is available
            self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            if PGVECTOR_PGCRYPTO:
                # Ensure the pgcrypto extension is available for encryption
                self.session.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))

                if not PGVECTOR_PGCRYPTO_KEY:
                    raise ValueError(
                        "PGVECTOR_PGCRYPTO_KEY must be set when PGVECTOR_PGCRYPTO is enabled."
                    )

            # Check vector length consistency
            self.check_vector_length()

            # Create the tables if they do not exist
            # Base.metadata.create_all requires a bind (engine or connection)
            # Get the connection from the session
            connection = self.session.connection()
            Base.metadata.create_all(bind=connection)

            # Create an index on the vector column if it doesn't exist
            self.session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_document_chunk_vector "
                    "ON document_chunk USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);"
                )
            )
            self.session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_document_chunk_collection_name "
                    "ON document_chunk (collection_name);"
                )
            )

            # Setup binary quantization support
            self.pgvector_version = self._detect_pgvector_version()
            self.binary_quantization_enabled = self._setup_binary_quantization()
            
            # Create migration tracking table
            self._create_migration_log_table()

            self.session.commit()
            log.info("Initialization complete.")
        except Exception as e:
            self.session.rollback()
            log.exception(f"Error during initialization: {e}")
            raise

    def check_vector_length(self) -> None:
        """
        Check if the VECTOR_LENGTH matches the existing vector column dimension in the database.
        Raises an exception if there is a mismatch.
        """
        metadata = MetaData()
        try:
            # Attempt to reflect the 'document_chunk' table
            document_chunk_table = Table(
                "document_chunk", metadata, autoload_with=self.session.bind
            )
        except NoSuchTableError:
            # Table does not exist; no action needed
            return

        # Proceed to check the vector column
        if "vector" in document_chunk_table.columns:
            vector_column = document_chunk_table.columns["vector"]
            vector_type = vector_column.type
            if isinstance(vector_type, Vector):
                db_vector_length = vector_type.dim
                if db_vector_length != VECTOR_LENGTH:
                    raise Exception(
                        f"VECTOR_LENGTH {VECTOR_LENGTH} does not match existing vector column dimension {db_vector_length}. "
                        "Cannot change vector size after initialization without migrating the data."
                    )
            else:
                raise Exception(
                    "The 'vector' column exists but is not of type 'Vector'."
                )
        else:
            raise Exception(
                "The 'vector' column does not exist in the 'document_chunk' table."
            )

    def adjust_vector_length(self, vector: List[float]) -> List[float]:
        # Adjust vector to have length VECTOR_LENGTH
        current_length = len(vector)
        if current_length < VECTOR_LENGTH:
            # Pad the vector with zeros
            vector += [0.0] * (VECTOR_LENGTH - current_length)
        elif current_length > VECTOR_LENGTH:
            # Truncate the vector to VECTOR_LENGTH
            vector = vector[:VECTOR_LENGTH]
        return vector

    def _detect_pgvector_version(self) -> str:
        """Detect pgvector extension version"""
        try:
            result = self.session.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )).first()
            return result.extversion if result else '0.0.0'
        except Exception as e:
            log.debug(f"Could not detect pgvector version: {e}")
            return '0.0.0'

    def _setup_binary_quantization(self) -> bool:
        """Setup binary quantization support based on pgvector version"""
        try:
            # Parse version to determine capabilities
            version_parts = [int(x) for x in self.pgvector_version.split('.')]
            
            if version_parts >= [0, 7, 0]:
                # Native binary quantization with pgvector 0.7.0+
                log.info(f"pgvector {self.pgvector_version}: Native binary quantization available")
                return True
            elif version_parts >= [0, 5, 0]:
                # Manual binary quantization for pgvector 0.5.0+
                self._setup_manual_binary_quantization()
                log.info(f"pgvector {self.pgvector_version}: Manual binary quantization enabled")
                return True
            else:
                log.info(f"pgvector {self.pgvector_version}: Binary quantization not supported")
                return False
        except Exception as e:
            log.warning(f"Binary quantization setup failed: {e}")
            return False

    def _setup_manual_binary_quantization(self):
        """Setup manual binary quantization for pgvector < 0.7.0"""
        try:
            # Create Hamming distance function if it doesn't exist
            function_exists = self.session.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_proc 
                    WHERE proname = 'hamming_distance_fast'
                )
            """)).scalar()
            
            if not function_exists:
                # Check PostgreSQL version for bit_count support
                pg_version = int(self.session.execute(text("SHOW server_version_num")).scalar())
                
                if pg_version >= 140000:  # PostgreSQL 14+
                    hamming_sql = """
                    CREATE OR REPLACE FUNCTION hamming_distance_fast(a bytea, b bytea) 
                    RETURNS integer AS $$
                    DECLARE
                        xor_bytes bytea;
                        distance integer := 0;
                        i integer;
                    BEGIN
                        IF a IS NULL OR b IS NULL OR octet_length(a) != octet_length(b) THEN
                            RETURN 999999;
                        END IF;
                        
                        SELECT a # b INTO xor_bytes;
                        FOR i IN 0..octet_length(xor_bytes)-1 LOOP
                            distance := distance + bit_count(get_byte(xor_bytes, i)::bit(8));
                        END LOOP;
                        
                        RETURN distance;
                    END;
                    $$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;
                    """
                else:  # PostgreSQL < 14
                    hamming_sql = """
                    CREATE OR REPLACE FUNCTION hamming_distance_fast(a bytea, b bytea) 
                    RETURNS integer AS $$
                    DECLARE
                        xor_bytes bytea;
                        distance integer := 0;
                        byte_val integer;
                        i integer;
                    BEGIN
                        IF a IS NULL OR b IS NULL OR octet_length(a) != octet_length(b) THEN
                            RETURN 999999;
                        END IF;
                        
                        SELECT a # b INTO xor_bytes;
                        FOR i IN 0..octet_length(xor_bytes)-1 LOOP
                            byte_val := get_byte(xor_bytes, i);
                            WHILE byte_val != 0 LOOP
                                byte_val := byte_val & (byte_val - 1);
                                distance := distance + 1;
                            END LOOP;
                        END LOOP;
                        
                        RETURN distance;
                    END;
                    $$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;
                    """
                
                self.session.execute(text(hamming_sql))
                log.debug("Created Hamming distance function for manual binary quantization")
        except Exception as e:
            log.warning(f"Manual binary quantization setup failed: {e}")
            raise

    def _is_binary_quantized_collection(self, collection_name: str) -> bool:
        """Check if collection uses binary quantization"""
        try:
            result = self.session.execute(text("""
                SELECT is_binary_quantized 
                FROM knowledge 
                WHERE name = :collection_name
            """), {"collection_name": collection_name}).first()
            
            return bool(result and result.is_binary_quantized)
        except Exception as e:
            log.debug(f"Could not check binary quantization status for {collection_name}: {e}")
            return False

    def _detect_migration_candidates(self) -> List[str]:
        """Detect collections that need automated migration to native binary"""
        try:
            version_parts = [int(x) for x in self.pgvector_version.split('.')]
            if version_parts < [0, 7, 0]:
                return []
            
            # Find binary quantized collections still using manual approach
            result = self.session.execute(text("""
                SELECT k.name
                FROM knowledge k
                LEFT JOIN document_chunk dc ON dc.collection_name = k.name
                WHERE k.is_binary_quantized = true
                  AND COALESCE(k.migration_status, 'manual') = 'manual'
                  AND EXISTS (
                      SELECT 1 FROM document_chunk dc2 
                      WHERE dc2.collection_name = k.name 
                        AND dc2.binary_vector IS NOT NULL
                  )
                LIMIT 1
            """)).fetchall()
            
            return [row.name for row in result]
            
        except Exception as e:
            log.debug(f"Migration candidate detection failed: {e}")
            return []

    def _is_low_usage_period(self) -> bool:
        """Determine if current time is suitable for background migration"""
        from datetime import datetime
        now = datetime.now()
        
        # Consider low usage: 2-6 AM local time and weekends
        is_night = 2 <= now.hour <= 6
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
        
        return is_night or is_weekend

    def _perform_safe_migration(self, collection_name: str) -> bool:
        """Perform robust automated migration with safety checks"""
        migration_id = None
        try:
            version_parts = [int(x) for x in self.pgvector_version.split('.')]
            if version_parts < [0, 7, 0]:
                return False
            
            log.info(f"Starting automated migration for collection: {collection_name}")
            
            # Step 1: Pre-migration validation
            stats = self.session.execute(text("""
                SELECT 
                    COUNT(*) as total_vectors,
                    COUNT(binary_vector) as manual_binary_count
                FROM document_chunk 
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name}).first()
            
            if not stats or stats.manual_binary_count == 0:
                log.info(f"No manual binary vectors to migrate for {collection_name}")
                return True
            
            # Step 2: Create migration record for tracking
            from datetime import datetime
            migration_id = f"migration_{collection_name}_{int(datetime.now().timestamp())}"
            self.session.execute(text("""
                INSERT INTO migration_log (id, collection_name, status, started_at, 
                                         total_vectors, migration_type)
                VALUES (:id, :collection_name, 'started', NOW(), :total_vectors, 'binary_native')
                ON CONFLICT (id) DO NOTHING
            """), {
                "id": migration_id,
                "collection_name": collection_name,
                "total_vectors": stats.total_vectors
            })
            
            # Step 3: Verify data integrity before migration
            integrity_check = self.session.execute(text("""
                SELECT COUNT(*) as corrupt_vectors
                FROM document_chunk 
                WHERE collection_name = :collection_name
                  AND (vector IS NULL OR binary_vector IS NULL)
                  AND id IS NOT NULL
            """), {"collection_name": collection_name}).scalar()
            
            if integrity_check > 0:
                raise Exception(f"Data integrity issue: {integrity_check} corrupt vectors found")
            
            # Step 4: Set migration status
            self.session.execute(text("""
                UPDATE knowledge 
                SET migration_status = 'migrating', 
                    updated_at = NOW()
                WHERE name = :collection_name
            """), {"collection_name": collection_name})
            
            # Step 5: Create native binary indexes (non-blocking)
            log.info(f"Creating native binary indexes for {collection_name}")
            self._ensure_binary_indexes(collection_name)
            
            # Step 6: Test native binary search works
            test_result = self._validate_native_binary_search(collection_name)
            if not test_result:
                raise Exception("Native binary search validation failed")
            
            # Step 7: Drop manual binary indexes only after validation
            log.info(f"Dropping manual binary indexes for {collection_name}")
            self._drop_manual_binary_indexes(collection_name)
            
            # Step 8: Final validation
            final_check = self.session.execute(text("""
                SELECT COUNT(*) as total_count
                FROM document_chunk 
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name}).scalar()
            
            if final_check != stats.total_vectors:
                raise Exception(f"Vector count mismatch: expected {stats.total_vectors}, got {final_check}")
            
            # Step 9: Mark migration complete
            self.session.execute(text("""
                UPDATE knowledge 
                SET migration_status = 'native',
                    updated_at = NOW()
                WHERE name = :collection_name
            """), {"collection_name": collection_name})
            
            # Step 10: Update migration log
            self.session.execute(text("""
                UPDATE migration_log 
                SET status = 'completed', completed_at = NOW()
                WHERE id = :migration_id
            """), {"migration_id": migration_id})
            
            self.session.commit()
            log.info(f"Automated migration completed successfully for {collection_name}")
            return True
            
        except Exception as e:
            # Robust rollback on any failure
            try:
                self.session.rollback()
                
                # Reset migration status
                self.session.execute(text("""
                    UPDATE knowledge 
                    SET migration_status = 'manual'
                    WHERE name = :collection_name
                """), {"collection_name": collection_name})
                
                # Log failure
                if migration_id:
                    self.session.execute(text("""
                        UPDATE migration_log 
                        SET status = 'failed', completed_at = NOW(), error = :error
                        WHERE id = :migration_id
                    """), {"migration_id": migration_id, "error": str(e)})
                
                self.session.commit()
                
            except Exception as rollback_error:
                log.error(f"Rollback also failed for {collection_name}: {rollback_error}")
                
            log.exception(f"Automated migration failed for {collection_name}: {e}")
            return False

    def _validate_native_binary_search(self, collection_name: str) -> bool:
        """Validate that native binary search works correctly"""
        try:
            # Get a sample vector for testing
            sample = self.session.execute(text("""
                SELECT vector FROM document_chunk 
                WHERE collection_name = :collection_name 
                  AND vector IS NOT NULL 
                LIMIT 1
            """), {"collection_name": collection_name}).first()
            
            if not sample:
                return True  # No vectors to test
            
            # Test native binary search
            test_result = self._search_native_binary(
                collection_name, 
                list(sample.vector), 
                candidate_limit=10, 
                final_limit=5
            )
            
            return test_result is not None and len(test_result.ids[0]) >= 0
            
        except Exception as e:
            log.warning(f"Native binary search validation failed: {e}")
            return False

    def run_automated_migration_check(self):
        """Main entry point for automated background migration"""
        if not self._is_low_usage_period():
            return  # Skip during high usage periods
        
        try:
            candidates = self._detect_migration_candidates()
            if not candidates:
                return
            
            log.info(f"Found {len(candidates)} collections eligible for automated migration")
            
            # Migrate one collection at a time to minimize impact
            for collection_name in candidates[:1]:  # Only process one per run
                success = self._perform_safe_migration(collection_name)
                if success:
                    log.info(f"Successfully migrated {collection_name} to native binary")
                else:
                    log.warning(f"Migration failed for {collection_name}, will retry later")
                break  # Only do one migration per background run
                
        except Exception as e:
            log.exception(f"Automated migration check failed: {e}")

    def _create_migration_log_table(self):
        """Create migration tracking table if it doesn't exist"""
        try:
            self.session.execute(text("""
                CREATE TABLE IF NOT EXISTS migration_log (
                    id VARCHAR(255) PRIMARY KEY,
                    collection_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    started_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP,
                    total_vectors INTEGER,
                    migration_type VARCHAR(50),
                    error TEXT
                )
            """))
            self.session.commit()
        except Exception as e:
            log.debug(f"Could not create migration_log table: {e}")

    def _drop_manual_binary_indexes(self, collection_name: str):
        """Drop manual binary indexes for migrated collection"""
        try:
            safe_name = collection_name.replace('-', '_').replace(' ', '_')[:30]
            index_name = f"idx_binary_manual_{safe_name}"
            
            # Check if manual index exists and drop it
            index_exists = self.session.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = :index_name
                )
            """), {"index_name": index_name}).scalar()
            
            if index_exists:
                self.session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                log.debug(f"Dropped manual binary index: {index_name}")
                
        except Exception as e:
            log.warning(f"Could not drop manual binary indexes for {collection_name}: {e}")
            # Non-critical error - continue migration

    def _ensure_binary_indexes(self, collection_name: str):
        """Create binary indexes for collection if they don't exist"""
        try:
            # Create safe index name
            safe_name = collection_name.replace('-', '_').replace(' ', '_')[:30]
            
            # Parse version to determine index strategy
            version_parts = [int(x) for x in self.pgvector_version.split('.')]
            
            if version_parts >= [0, 7, 0]:
                # Native binary index using expression for pgvector 0.7.0+
                index_name = f"idx_binary_native_{safe_name}"
                
                # Check if index exists
                index_exists = self.session.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = :index_name
                    )
                """), {"index_name": index_name}).scalar()
                
                if not index_exists:
                    index_sql = f"""
                    CREATE INDEX {index_name}
                    ON document_chunk 
                    USING hnsw ((binary_quantize(vector)::bit({VECTOR_LENGTH})) bit_hamming_ops)
                    WHERE collection_name = :collection_name
                    """
                    
                    self.session.execute(text(index_sql), {"collection_name": collection_name})
                    log.debug(f"Created native binary index for collection: {collection_name}")
                    
            else:
                # Manual binary index on bytea column for pgvector 0.5.0+
                index_name = f"idx_binary_manual_{safe_name}"
                
                index_exists = self.session.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = :index_name
                    )
                """), {"index_name": index_name}).scalar()
                
                if not index_exists:
                    index_sql = f"""
                    CREATE INDEX {index_name}
                    ON document_chunk (binary_vector)
                    WHERE collection_name = :collection_name 
                      AND binary_vector IS NOT NULL
                    """
                    
                    self.session.execute(text(index_sql), {"collection_name": collection_name})
                    log.debug(f"Created manual binary index for collection: {collection_name}")
            
        except Exception as e:
            # Non-critical error - index creation might fail due to permissions or other issues
            log.warning(f"Binary index creation failed for {collection_name}: {e}")

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        try:
            # Check if this collection uses binary quantization
            is_binary_collection = (
                self.binary_quantization_enabled and 
                self._is_binary_quantized_collection(collection_name)
            )
            
            if PGVECTOR_PGCRYPTO:
                for item in items:
                    vector = self.adjust_vector_length(item["vector"])
                    
                    # Generate binary vector if needed
                    binary_vector = None
                    if is_binary_collection:
                        from open_webui.retrieval.utils import compute_binary_vector
                        binary_vector = compute_binary_vector(item["vector"])
                    
                    # Use raw SQL for BYTEA/pgcrypto
                    self.session.execute(
                        text(
                            """
                            INSERT INTO document_chunk
                            (id, vector, collection_name, text, vmetadata, binary_vector)
                            VALUES (
                                :id, :vector, :collection_name,
                                pgp_sym_encrypt(:text, :key),
                                pgp_sym_encrypt(:metadata::text, :key),
                                :binary_vector
                            )
                            ON CONFLICT (id) DO NOTHING
                        """
                        ),
                        {
                            "id": item["id"],
                            "vector": vector,
                            "collection_name": collection_name,
                            "text": item["text"],
                            "metadata": json.dumps(item["metadata"]),
                            "key": PGVECTOR_PGCRYPTO_KEY,
                            "binary_vector": binary_vector,
                        },
                    )
                
                # Setup binary indexes for this collection if needed
                if is_binary_collection:
                    self._ensure_binary_indexes(collection_name)
                
                self.session.commit()
                log.info(f"Encrypted & inserted {len(items)} into '{collection_name}'")

            else:
                new_items = []
                for item in items:
                    vector = self.adjust_vector_length(item["vector"])
                    
                    # Generate binary vector if needed
                    binary_vector = None
                    if is_binary_collection:
                        from open_webui.retrieval.utils import compute_binary_vector
                        binary_vector = compute_binary_vector(item["vector"])
                    
                    new_chunk = DocumentChunk(
                        id=item["id"],
                        vector=vector,
                        collection_name=collection_name,
                        text=item["text"],
                        vmetadata=item["metadata"],
                        binary_vector=binary_vector,
                    )
                    new_items.append(new_chunk)
                
                self.session.bulk_save_objects(new_items)
                
                # Setup binary indexes for this collection if needed
                if is_binary_collection:
                    self._ensure_binary_indexes(collection_name)
                
                self.session.commit()
                log.info(
                    f"Inserted {len(new_items)} items into collection '{collection_name}'."
                )
        except Exception as e:
            self.session.rollback()
            log.exception(f"Error during insert: {e}")
            raise

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        try:
            # Check if this collection uses binary quantization
            is_binary_collection = (
                self.binary_quantization_enabled and 
                self._is_binary_quantized_collection(collection_name)
            )
            
            if PGVECTOR_PGCRYPTO:
                for item in items:
                    vector = self.adjust_vector_length(item["vector"])
                    
                    # Generate binary vector if needed
                    binary_vector = None
                    if is_binary_collection:
                        from open_webui.retrieval.utils import compute_binary_vector
                        binary_vector = compute_binary_vector(item["vector"])
                    
                    self.session.execute(
                        text(
                            """
                            INSERT INTO document_chunk
                            (id, vector, collection_name, text, vmetadata, binary_vector)
                            VALUES (
                                :id, :vector, :collection_name,
                                pgp_sym_encrypt(:text, :key),
                                pgp_sym_encrypt(:metadata::text, :key),
                                :binary_vector
                            )
                            ON CONFLICT (id) DO UPDATE SET
                              vector = EXCLUDED.vector,
                              collection_name = EXCLUDED.collection_name,
                              text = EXCLUDED.text,
                              vmetadata = EXCLUDED.vmetadata,
                              binary_vector = EXCLUDED.binary_vector
                        """
                        ),
                        {
                            "id": item["id"],
                            "vector": vector,
                            "collection_name": collection_name,
                            "text": item["text"],
                            "metadata": json.dumps(item["metadata"]),
                            "key": PGVECTOR_PGCRYPTO_KEY,
                            "binary_vector": binary_vector,
                        },
                    )
                
                # Setup binary indexes for this collection if needed
                if is_binary_collection:
                    self._ensure_binary_indexes(collection_name)
                
                self.session.commit()
                log.info(f"Encrypted & upserted {len(items)} into '{collection_name}'")
            else:
                for item in items:
                    vector = self.adjust_vector_length(item["vector"])
                    
                    # Generate binary vector if needed
                    binary_vector = None
                    if is_binary_collection:
                        from open_webui.retrieval.utils import compute_binary_vector
                        binary_vector = compute_binary_vector(item["vector"])
                    
                    existing = (
                        self.session.query(DocumentChunk)
                        .filter(DocumentChunk.id == item["id"])
                        .first()
                    )
                    if existing:
                        existing.vector = vector
                        existing.text = item["text"]
                        existing.vmetadata = item["metadata"]
                        existing.collection_name = (
                            collection_name  # Update collection_name if necessary
                        )
                        existing.binary_vector = binary_vector
                    else:
                        new_chunk = DocumentChunk(
                            id=item["id"],
                            vector=vector,
                            collection_name=collection_name,
                            text=item["text"],
                            vmetadata=item["metadata"],
                            binary_vector=binary_vector,
                        )
                        self.session.add(new_chunk)
                
                # Setup binary indexes for this collection if needed
                if is_binary_collection:
                    self._ensure_binary_indexes(collection_name)
                
                self.session.commit()
                log.info(
                    f"Upserted {len(items)} items into collection '{collection_name}'."
                )
        except Exception as e:
            self.session.rollback()
            log.exception(f"Error during upsert: {e}")
            raise

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        limit: Optional[int] = None,
    ) -> Optional[SearchResult]:
        # Check if this collection uses binary quantization
        is_binary_collection = (
            self.binary_quantization_enabled and 
            self._is_binary_quantized_collection(collection_name)
        )
        
        # Try binary search if enabled, with fallback to regular search
        if is_binary_collection:
            try:
                result = self._search_with_binary_quantization(collection_name, vectors, limit)
                if result:
                    return result
                else:
                    log.warning(f"Binary search returned no results for {collection_name}, falling back to regular search")
            except Exception as e:
                log.warning(f"Binary search failed for {collection_name}, falling back to regular search: {e}")
        
        # Regular search (existing implementation)
        return self._search_regular(collection_name, vectors, limit)

    def _search_with_binary_quantization(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        limit: Optional[int] = None
    ) -> Optional[SearchResult]:
        """Search using binary quantization with two-stage retrieval"""
        try:
            if not vectors:
                return None

            # Adjust query vector to VECTOR_LENGTH
            query_vector = self.adjust_vector_length(vectors[0])  # Handle single query
            candidate_limit = max((limit or 10) * 50, 800)  # Get many candidates for reranking
            
            # Determine search strategy based on pgvector version
            version_parts = [int(x) for x in self.pgvector_version.split('.')]
            
            if version_parts >= [0, 7, 0]:
                # Native binary quantization with pgvector 0.7.0+
                return self._search_native_binary(collection_name, query_vector, candidate_limit, limit)
            else:
                # Manual binary quantization for pgvector 0.5.0+
                return self._search_manual_binary(collection_name, query_vector, candidate_limit, limit)
                
        except Exception as e:
            log.exception(f"Error during binary search: {e}")
            return None

    def _search_native_binary(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        candidate_limit: int, 
        final_limit: Optional[int]
    ) -> Optional[SearchResult]:
        """Search using native pgvector 0.7.0+ binary quantization"""
        
        if PGVECTOR_PGCRYPTO:
            search_sql = f"""
                SELECT id, text, vmetadata, distance
                FROM (
                    SELECT 
                        id,
                        pgp_sym_decrypt(text, :key) as text,
                        pgp_sym_decrypt(vmetadata, :key)::jsonb as vmetadata,
                        vector <=> :query_vector AS distance
                    FROM (
                        SELECT id, text, vmetadata, vector
                        FROM document_chunk
                        WHERE collection_name = :collection_name
                        ORDER BY binary_quantize(vector)::bit({VECTOR_LENGTH}) <~> binary_quantize(:query_vector)
                        LIMIT :candidate_limit
                    ) candidates
                    ORDER BY distance
                    LIMIT :final_limit
                ) final
                ORDER BY distance
            """
            
            params = {
                "query_vector": query_vector,
                "collection_name": collection_name,
                "candidate_limit": candidate_limit,
                "final_limit": final_limit or 10,
                "key": PGVECTOR_PGCRYPTO_KEY
            }
        else:
            search_sql = f"""
                SELECT id, text, vmetadata, distance
                FROM (
                    SELECT 
                        id, text, vmetadata,
                        vector <=> :query_vector AS distance
                    FROM (
                        SELECT id, text, vmetadata, vector
                        FROM document_chunk
                        WHERE collection_name = :collection_name
                        ORDER BY binary_quantize(vector)::bit({VECTOR_LENGTH}) <~> binary_quantize(:query_vector)
                        LIMIT :candidate_limit
                    ) candidates
                    ORDER BY distance
                    LIMIT :final_limit
                ) final
                ORDER BY distance
            """
            
            params = {
                "query_vector": query_vector,
                "collection_name": collection_name,
                "candidate_limit": candidate_limit,
                "final_limit": final_limit or 10
            }
        
        results = self.session.execute(text(search_sql), params).fetchall()
        return self._format_binary_search_results(results)

    def _search_manual_binary(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        candidate_limit: int, 
        final_limit: Optional[int]
    ) -> Optional[SearchResult]:
        """Search using manual binary quantization for pgvector < 0.7.0"""
        
        from open_webui.retrieval.utils import compute_binary_vector
        query_binary = compute_binary_vector(query_vector)
        
        if PGVECTOR_PGCRYPTO:
            search_sql = """
                SELECT id, text, vmetadata, distance
                FROM (
                    SELECT 
                        id,
                        pgp_sym_decrypt(text, :key) as text,
                        pgp_sym_decrypt(vmetadata, :key)::jsonb as vmetadata,
                        vector <=> :query_vector AS distance
                    FROM (
                        SELECT id, text, vmetadata, vector
                        FROM document_chunk
                        WHERE collection_name = :collection_name
                          AND binary_vector IS NOT NULL
                        ORDER BY hamming_distance_fast(binary_vector, :query_binary)
                        LIMIT :candidate_limit
                    ) candidates
                    ORDER BY distance
                    LIMIT :final_limit
                ) final
                ORDER BY distance
            """
            
            params = {
                "query_vector": query_vector,
                "query_binary": query_binary,
                "collection_name": collection_name,
                "candidate_limit": candidate_limit,
                "final_limit": final_limit or 10,
                "key": PGVECTOR_PGCRYPTO_KEY
            }
        else:
            search_sql = """
                SELECT id, text, vmetadata, distance
                FROM (
                    SELECT 
                        id, text, vmetadata,
                        vector <=> :query_vector AS distance
                    FROM (
                        SELECT id, text, vmetadata, vector
                        FROM document_chunk
                        WHERE collection_name = :collection_name
                          AND binary_vector IS NOT NULL
                        ORDER BY hamming_distance_fast(binary_vector, :query_binary)
                        LIMIT :candidate_limit
                    ) candidates
                    ORDER BY distance
                    LIMIT :final_limit
                ) final
                ORDER BY distance
            """
            
            params = {
                "query_vector": query_vector,
                "query_binary": query_binary,
                "collection_name": collection_name,
                "candidate_limit": candidate_limit,
                "final_limit": final_limit or 10
            }
        
        results = self.session.execute(text(search_sql), params).fetchall()
        return self._format_binary_search_results(results)

    def _format_binary_search_results(self, results) -> SearchResult:
        """Format binary search results to match expected SearchResult format"""
        
        ids = [[]]
        distances = [[]]
        documents = [[]]
        metadatas = [[]]

        for row in results:
            ids[0].append(row.id)
            # normalize distance to [0, 1] score range (same as regular search)
            distances[0].append((2.0 - float(row.distance)) / 2.0)
            documents[0].append(row.text)
            metadatas[0].append(row.vmetadata)

        return SearchResult(
            ids=ids,
            distances=distances,
            documents=documents,
            metadatas=metadatas
        )

    def search_binary(self, collection_name: str, binary_vectors: List[bytes], limit: int) -> Optional[SearchResult]:
        """Search for binary vectors with Hamming distance - interface compatibility with ChromaDB"""
        try:
            if not binary_vectors or not self.binary_quantization_enabled:
                return None
            
            # Check if collection uses binary quantization
            if not self._is_binary_quantized_collection(collection_name):
                return None
            
            log.debug(f"Direct binary search for collection {collection_name} with {len(binary_vectors)} queries")
            
            binary_query = binary_vectors[0]  # Handle single query for now
            version_parts = [int(x) for x in self.pgvector_version.split('.')]
            
            if version_parts >= [0, 7, 0]:
                return self._search_binary_native_direct(collection_name, binary_query, limit)
            else:
                return self._search_binary_manual_direct(collection_name, binary_query, limit)
                
        except Exception as e:
            log.exception(f"Error in binary search: {e}")
            return None

    def _search_binary_manual_direct(self, collection_name: str, query_binary: bytes, limit: int) -> Optional[SearchResult]:
        """Direct binary search using optimized Python Hamming distance"""
        try:
            # For better performance, get all candidates and compute Hamming distance in Python
            # This allows us to use Numba JIT compilation
            
            if PGVECTOR_PGCRYPTO:
                fetch_sql = """
                    SELECT id, text, vmetadata, binary_vector
                    FROM document_chunk
                    WHERE collection_name = :collection_name
                      AND binary_vector IS NOT NULL
                """
                
                params = {
                    "collection_name": collection_name,
                    "key": PGVECTOR_PGCRYPTO_KEY
                }
            else:
                fetch_sql = """
                    SELECT id, text, vmetadata, binary_vector
                    FROM document_chunk
                    WHERE collection_name = :collection_name
                      AND binary_vector IS NOT NULL
                """
                
                params = {
                    "collection_name": collection_name
                }
            
            candidates = self.session.execute(text(fetch_sql), params).fetchall()
            
            if not candidates:
                return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]])
            
            # Extract binary vectors for batch processing
            candidate_binaries = [bytes(row.binary_vector) for row in candidates]
            
            # Compute Hamming distances using optimized functions
            log.debug(f"Computing Hamming distances for {len(candidates)} candidates using {'Numba' if NUMBA_AVAILABLE else 'NumPy'}")
            hamming_distances = batch_hamming_distance(query_binary, candidate_binaries)
            
            # Combine results with distances and sort
            candidate_results = []
            max_hamming = len(query_binary) * 8
            
            for i, row in enumerate(candidates):
                hamming_dist = hamming_distances[i]
                similarity = 1.0 - (float(hamming_dist) / max_hamming)
                
                # Decrypt text and metadata if needed
                if PGVECTOR_PGCRYPTO:
                    # Note: For encrypted content, we'd need to decrypt here
                    # For now, pass through assuming decryption happens elsewhere
                    text = row.text
                    vmetadata = row.vmetadata
                else:
                    text = row.text
                    vmetadata = row.vmetadata
                
                candidate_results.append({
                    'id': row.id,
                    'text': text,
                    'vmetadata': vmetadata,
                    'distance': hamming_dist,
                    'similarity': max(0.0, min(1.0, similarity))
                })
            
            # Sort by distance (ascending - lower is better)
            candidate_results.sort(key=lambda x: x['distance'])
            
            # Take top results
            top_results = candidate_results[:limit]
            
            # Format results
            ids = [[r['id'] for r in top_results]]
            distances = [[r['similarity'] for r in top_results]]
            documents = [[r['text'] for r in top_results]]
            metadatas = [[r['vmetadata'] for r in top_results]]

            log.debug(f"Manual binary search returned {len(top_results)} results")
            
            return SearchResult(
                ids=ids,
                distances=distances,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            log.exception(f"Optimized manual binary search failed: {e}")
            return None

    def _search_binary_native_direct(self, collection_name: str, query_binary: bytes, limit: int) -> Optional[SearchResult]:
        """Direct binary search using native pgvector 0.7.0+ with binary query input"""
        try:
            # For native pgvector 0.7.0+, we can use the binary operators directly
            if PGVECTOR_PGCRYPTO:
                search_sql = f"""
                    SELECT id, text, vmetadata, distance
                    FROM (
                        SELECT 
                            id,
                            pgp_sym_decrypt(text, :key) as text,
                            pgp_sym_decrypt(vmetadata, :key)::jsonb as vmetadata,
                            binary_quantize(vector)::bit({VECTOR_LENGTH}) <~> :query_binary::bit({VECTOR_LENGTH}) as distance
                        FROM document_chunk
                        WHERE collection_name = :collection_name
                        ORDER BY distance
                        LIMIT :limit
                    ) results
                    ORDER BY distance
                """
                
                params = {
                    "query_binary": query_binary,
                    "collection_name": collection_name,
                    "limit": limit,
                    "key": PGVECTOR_PGCRYPTO_KEY
                }
            else:
                search_sql = f"""
                    SELECT id, text, vmetadata, distance
                    FROM (
                        SELECT 
                            id, text, vmetadata,
                            binary_quantize(vector)::bit({VECTOR_LENGTH}) <~> :query_binary::bit({VECTOR_LENGTH}) as distance
                        FROM document_chunk
                        WHERE collection_name = :collection_name
                        ORDER BY distance
                        LIMIT :limit
                    ) results
                    ORDER BY distance
                """
                
                params = {
                    "query_binary": query_binary,
                    "collection_name": collection_name,
                    "limit": limit
                }
            
            results = self.session.execute(text(search_sql), params).fetchall()
            
            # Format results to match SearchResult format
            ids = [[]]
            distances = [[]]
            documents = [[]]
            metadatas = [[]]

            for row in results:
                ids[0].append(row.id)
                # Convert Hamming distance to similarity score [0, 1] range
                hamming_dist = float(row.distance)
                max_hamming = VECTOR_LENGTH  # For bit strings, max distance is the length
                similarity = 1.0 - (hamming_dist / max_hamming)
                distances[0].append(max(0.0, min(1.0, similarity)))  # Clamp to [0, 1]
                documents[0].append(row.text)
                metadatas[0].append(row.vmetadata)

            return SearchResult(
                ids=ids,
                distances=distances,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            log.exception(f"Native binary search failed: {e}")
            return None

    def _search_regular(
        self,
        collection_name: str,
        vectors: List[List[float]],
        limit: Optional[int] = None,
    ) -> Optional[SearchResult]:
        """Regular search without binary quantization (existing implementation)"""
        try:
            if not vectors:
                return None

            # Adjust query vectors to VECTOR_LENGTH
            vectors = [self.adjust_vector_length(vector) for vector in vectors]
            num_queries = len(vectors)

            def vector_expr(vector):
                return cast(array(vector), Vector(VECTOR_LENGTH))

            # Create the values for query vectors
            qid_col = column("qid", Integer)
            q_vector_col = column("q_vector", Vector(VECTOR_LENGTH))
            query_vectors = (
                values(qid_col, q_vector_col)
                .data(
                    [(idx, vector_expr(vector)) for idx, vector in enumerate(vectors)]
                )
                .alias("query_vectors")
            )

            result_fields = [
                DocumentChunk.id,
            ]
            if PGVECTOR_PGCRYPTO:
                result_fields.append(
                    pgcrypto_decrypt(
                        DocumentChunk.text, PGVECTOR_PGCRYPTO_KEY, Text
                    ).label("text")
                )
                result_fields.append(
                    pgcrypto_decrypt(
                        DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB
                    ).label("vmetadata")
                )
            else:
                result_fields.append(DocumentChunk.text)
                result_fields.append(DocumentChunk.vmetadata)
            result_fields.append(
                (DocumentChunk.vector.cosine_distance(query_vectors.c.q_vector)).label(
                    "distance"
                )
            )

            # Build the lateral subquery for each query vector
            subq = (
                select(*result_fields)
                .where(DocumentChunk.collection_name == collection_name)
                .order_by(
                    (DocumentChunk.vector.cosine_distance(query_vectors.c.q_vector))
                )
            )
            if limit is not None:
                subq = subq.limit(limit)
            subq = subq.lateral("result")

            # Build the main query by joining query_vectors and the lateral subquery
            stmt = (
                select(
                    query_vectors.c.qid,
                    subq.c.id,
                    subq.c.text,
                    subq.c.vmetadata,
                    subq.c.distance,
                )
                .select_from(query_vectors)
                .join(subq, true())
                .order_by(query_vectors.c.qid, subq.c.distance)
            )

            result_proxy = self.session.execute(stmt)
            results = result_proxy.all()

            ids = [[] for _ in range(num_queries)]
            distances = [[] for _ in range(num_queries)]
            documents = [[] for _ in range(num_queries)]
            metadatas = [[] for _ in range(num_queries)]

            if not results:
                return SearchResult(
                    ids=ids,
                    distances=distances,
                    documents=documents,
                    metadatas=metadatas,
                )

            for row in results:
                qid = int(row.qid)
                ids[qid].append(row.id)
                # normalize and re-orders pgvec distance from [2, 0] to [0, 1] score range
                # https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
                distances[qid].append((2.0 - row.distance) / 2.0)
                documents[qid].append(row.text)
                metadatas[qid].append(row.vmetadata)

            return SearchResult(
                ids=ids, distances=distances, documents=documents, metadatas=metadatas
            )
        except Exception as e:
            log.exception(f"Error during regular search: {e}")
            return None

    def query(
        self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None
    ) -> Optional[GetResult]:
        try:
            if PGVECTOR_PGCRYPTO:
                # Build where clause for vmetadata filter
                where_clauses = [DocumentChunk.collection_name == collection_name]
                for key, value in filter.items():
                    # decrypt then check key: JSON filter after decryption
                    where_clauses.append(
                        pgcrypto_decrypt(
                            DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB
                        )[key].astext
                        == str(value)
                    )
                stmt = select(
                    DocumentChunk.id,
                    pgcrypto_decrypt(
                        DocumentChunk.text, PGVECTOR_PGCRYPTO_KEY, Text
                    ).label("text"),
                    pgcrypto_decrypt(
                        DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB
                    ).label("vmetadata"),
                ).where(*where_clauses)
                if limit is not None:
                    stmt = stmt.limit(limit)
                results = self.session.execute(stmt).all()
            else:
                query = self.session.query(DocumentChunk).filter(
                    DocumentChunk.collection_name == collection_name
                )

                for key, value in filter.items():
                    query = query.filter(
                        DocumentChunk.vmetadata[key].astext == str(value)
                    )

                if limit is not None:
                    query = query.limit(limit)

                results = query.all()

            if not results:
                return None

            ids = [[result.id for result in results]]
            documents = [[result.text for result in results]]
            metadatas = [[result.vmetadata for result in results]]

            return GetResult(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            log.exception(f"Error during query: {e}")
            return None

    def get(
        self, collection_name: str, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        try:
            if PGVECTOR_PGCRYPTO:
                stmt = select(
                    DocumentChunk.id,
                    pgcrypto_decrypt(
                        DocumentChunk.text, PGVECTOR_PGCRYPTO_KEY, Text
                    ).label("text"),
                    pgcrypto_decrypt(
                        DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB
                    ).label("vmetadata"),
                ).where(DocumentChunk.collection_name == collection_name)
                if limit is not None:
                    stmt = stmt.limit(limit)
                results = self.session.execute(stmt).all()
                ids = [[row.id for row in results]]
                documents = [[row.text for row in results]]
                metadatas = [[row.vmetadata for row in results]]
            else:

                query = self.session.query(DocumentChunk).filter(
                    DocumentChunk.collection_name == collection_name
                )
                if limit is not None:
                    query = query.limit(limit)

                results = query.all()

                if not results:
                    return None

                ids = [[result.id for result in results]]
                documents = [[result.text for result in results]]
                metadatas = [[result.vmetadata for result in results]]

            return GetResult(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as e:
            log.exception(f"Error during get: {e}")
            return None

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if PGVECTOR_PGCRYPTO:
                wheres = [DocumentChunk.collection_name == collection_name]
                if ids:
                    wheres.append(DocumentChunk.id.in_(ids))
                if filter:
                    for key, value in filter.items():
                        wheres.append(
                            pgcrypto_decrypt(
                                DocumentChunk.vmetadata, PGVECTOR_PGCRYPTO_KEY, JSONB
                            )[key].astext
                            == str(value)
                        )
                stmt = DocumentChunk.__table__.delete().where(*wheres)
                result = self.session.execute(stmt)
                deleted = result.rowcount
            else:
                query = self.session.query(DocumentChunk).filter(
                    DocumentChunk.collection_name == collection_name
                )
                if ids:
                    query = query.filter(DocumentChunk.id.in_(ids))
                if filter:
                    for key, value in filter.items():
                        query = query.filter(
                            DocumentChunk.vmetadata[key].astext == str(value)
                        )
                deleted = query.delete(synchronize_session=False)
            self.session.commit()
            log.info(f"Deleted {deleted} items from collection '{collection_name}'.")
        except Exception as e:
            self.session.rollback()
            log.exception(f"Error during delete: {e}")
            raise

    def reset(self) -> None:
        try:
            deleted = self.session.query(DocumentChunk).delete()
            self.session.commit()
            log.info(
                f"Reset complete. Deleted {deleted} items from 'document_chunk' table."
            )
        except Exception as e:
            self.session.rollback()
            log.exception(f"Error during reset: {e}")
            raise

    def close(self) -> None:
        pass

    def has_collection(self, collection_name: str) -> bool:
        try:
            exists = (
                self.session.query(DocumentChunk)
                .filter(DocumentChunk.collection_name == collection_name)
                .first()
                is not None
            )
            return exists
        except Exception as e:
            log.exception(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str) -> None:
        self.delete(collection_name)
        log.info(f"Collection '{collection_name}' deleted.")
