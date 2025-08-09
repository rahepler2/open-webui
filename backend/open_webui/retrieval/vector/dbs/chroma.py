import numpy as np
import chromadb
import logging
from chromadb import Settings
from chromadb.utils.batch_utils import create_batches

from typing import Optional, List, Union
import base64

from open_webui.retrieval.vector.main import (
    VectorDBBase,
    VectorItem,
    SearchResult,
    GetResult,
)
from open_webui.config import (
    CHROMA_DATA_PATH,
    CHROMA_HTTP_HOST,
    CHROMA_HTTP_PORT,
    CHROMA_HTTP_HEADERS,
    CHROMA_HTTP_SSL,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    CHROMA_CLIENT_AUTH_PROVIDER,
    CHROMA_CLIENT_AUTH_CREDENTIALS,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class ChromaClient(VectorDBBase):
    def __init__(self):
        settings_dict = {
            "allow_reset": True,
            "anonymized_telemetry": False,
        }
        if CHROMA_CLIENT_AUTH_PROVIDER is not None:
            settings_dict["chroma_client_auth_provider"] = CHROMA_CLIENT_AUTH_PROVIDER
        if CHROMA_CLIENT_AUTH_CREDENTIALS is not None:
            settings_dict["chroma_client_auth_credentials"] = (
                CHROMA_CLIENT_AUTH_CREDENTIALS
            )

        if CHROMA_HTTP_HOST != "":
            self.client = chromadb.HttpClient(
                host=CHROMA_HTTP_HOST,
                port=CHROMA_HTTP_PORT,
                headers=CHROMA_HTTP_HEADERS,
                ssl=CHROMA_HTTP_SSL,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
                settings=Settings(**settings_dict),
            )
        else:
            self.client = chromadb.PersistentClient(
                path=CHROMA_DATA_PATH,
                settings=Settings(**settings_dict),
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )

    def has_collection(self, collection_name: str) -> bool:
        # Check if the collection exists based on the collection name.
        collection_names = self.client.list_collections()
        return collection_name in collection_names

    def delete_collection(self, collection_name: str):
        # Delete the collection based on the collection name.
        return self.client.delete_collection(name=collection_name)

    def search(
        self, collection_name: str, vectors: list[list[float | int]], limit: int
    ) -> Optional[SearchResult]:
        # Search for the nearest neighbor items based on the vectors and return 'limit' number of results.
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection:
                result = collection.query(
                    query_embeddings=vectors,
                    n_results=limit,
                )

                # chromadb has cosine distance, 2 (worst) -> 0 (best). Re-odering to 0 -> 1
                # https://docs.trychroma.com/docs/collections/configure cosine equation
                distances: list = result["distances"][0]
                distances = [2 - dist for dist in distances]
                distances = [[dist / 2 for dist in distances]]

                return SearchResult(
                    **{
                        "ids": result["ids"],
                        "distances": distances,
                        "documents": result["documents"],
                        "metadatas": result["metadatas"],
                    }
                )
            return None
        except Exception as e:
            return None

    def query(
        self, collection_name: str, filter: dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        # Query the items from the collection based on the filter.
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection:
                result = collection.get(
                    where=filter,
                    limit=limit,
                )

                return GetResult(
                    **{
                        "ids": [result["ids"]],
                        "documents": [result["documents"]],
                        "metadatas": [result["metadatas"]],
                    }
                )
            return None
        except:
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        # Get all the items in the collection.
        collection = self.client.get_collection(name=collection_name)
        if collection:
            result = collection.get()
            return GetResult(
                **{
                    "ids": [result["ids"]],
                    "documents": [result["documents"]],
                    "metadatas": [result["metadatas"]],
                }
            )
        return None

    def insert(self, collection_name: str, items: list[VectorItem]):
        # Insert the items into the collection, if the collection does not exist, it will be created.
        collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        ids = [item.id for item in items]
        documents = [item.text for item in items]
        embeddings = [item.vector for item in items]
        
        # Handle binary vectors in metadata
        metadatas = []
        for item in items:
            metadata = item.metadata if item.metadata else {}
            if item.binary_vector:
                metadata["binary_vector"] = base64.b64encode(item.binary_vector).decode('utf-8')
            metadatas.append(metadata)

        for batch in create_batches(
            api=self.client,
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        ):
            collection.add(*batch)

    def upsert(self, collection_name: str, items: list[VectorItem]):
        # Update the items in the collection, if the items are not present, insert them. If the collection does not exist, it will be created.
        collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        ids = [item.id for item in items]
        documents = [item.text for item in items]
        embeddings = [item.vector for item in items]
        
        # Handle binary vectors in metadata
        metadatas = []
        for item in items:
            metadata = item.metadata if item.metadata else {}
            if item.binary_vector:
                metadata["binary_vector"] = base64.b64encode(item.binary_vector).decode('utf-8')
            metadatas.append(metadata)

        collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        # Delete the items from the collection based on the ids.
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection:
                if ids:
                    collection.delete(ids=ids)
                elif filter:
                    collection.delete(where=filter)
        except Exception as e:
            # If collection doesn't exist, that's fine - nothing to delete
            log.debug(
                f"Attempted to delete from non-existent collection {collection_name}. Ignoring."
            )
            pass

    def reset(self):
        # Resets the database. This will delete all collections and item entries.
        return self.client.reset()

    def _compute_hamming_distance(self, binary_vec1: bytes, binary_vec2: bytes) -> int:
        """Compute Hamming distance using pure numpy."""
        try:
            arr1 = np.frombuffer(binary_vec1, dtype=np.uint8)
            arr2 = np.frombuffer(binary_vec2, dtype=np.uint8)
            
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
            
            xor_result = arr1 ^ arr2
            binary_bits = np.unpackbits(xor_result)
            return int(np.sum(binary_bits))
        except Exception as e:
            log.error(f"Error computing Hamming distance: {e}")
            return 999999

    def search_binary(self, collection_name: str, binary_vectors: List[bytes], limit: int) -> Optional[SearchResult]:
        """Search for binary vectors with Hamming distance."""
        try:
            collection = self.client.get_collection(name=collection_name)
            all_results = collection.get(include=["metadatas", "documents"])
            
            # Early exit if no binary vectors
            if not any("binary_vector" in (meta or {}) for meta in all_results["metadatas"]):
                return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]]) 
            
            candidates = []
            for query_binary in binary_vectors:
                for i, metadata in enumerate(all_results["metadatas"]):
                    if metadata and "binary_vector" in metadata:
                        # Decode base64-stored binary vector
                        doc_binary = base64.b64decode(metadata["binary_vector"])
                        
                        hamming_dist = self._compute_hamming_distance(query_binary, doc_binary)
                        candidates.append({
                            "id": all_results["ids"][i],
                            "document": all_results["documents"][i],
                            "metadata": metadata,
                            "distance": hamming_dist,
                        })
            
            candidates.sort(key=lambda x: x["distance"])
            top_candidates = candidates[:limit]
            
            if top_candidates:
                return SearchResult(
                    ids=[[c["id"] for c in top_candidates]],
                    distances=[[c["distance"] for c in top_candidates]],
                    documents=[[c["document"] for c in top_candidates]],
                    metadatas=[[c["metadata"] for c in top_candidates]],
                )
            return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]]) 
        except Exception as e:
            log.error(f"Error in binary search: {e}")
            return None

    def search_filtered(self, collection_name: str, vectors: List[List[Union[float, int]]], 
                    limit: int, filtered_ids: List[str]) -> Optional[SearchResult]:
        """Search for vectors with ID filter."""
        try:
            collection = self.client.get_collection(name=collection_name)
            candidate_results = collection.get(ids=filtered_ids, include=["metadatas", "documents", "embeddings"])
            
            if not candidate_results["embeddings"]:
                return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]]) 
            
            query_vector = np.array(vectors[0])
            candidate_embeddings = np.array(candidate_results["embeddings"])
            
            query_norm = query_vector / np.linalg.norm(query_vector)
            candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            similarities = np.dot(candidate_norms, query_norm)
            
            # Convert similarities to distances to match existing search() format
            distances = 2 - similarities  # Same transformation as existing search()
            distances = distances / 2
            
            sorted_indices = np.argsort(distances)[:limit]  # Sort by distance (ascending)
            
            return SearchResult(
                ids=[[candidate_results["ids"][i] for i in sorted_indices]],
                distances=[distances[sorted_indices].tolist()],
                documents=[[candidate_results["documents"][i] for i in sorted_indices]],
                metadatas=[[candidate_results["metadatas"][i] for i in sorted_indices]],
            )
        except Exception as e:
            log.error(f"Error in filtered search: {e}")
            return None
