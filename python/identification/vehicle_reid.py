"""
Vehicle Re-Identification using embedding-based matching.

Used as fallback when license plate OCR fails (occlusion, distance, angle).
Also used for cross-camera tracking — linking the same vehicle across zones.

Approach:
  - Extract a feature embedding from each vehicle crop using a CNN backbone
  - Store embeddings in a gallery (vehicle_id → embedding)
  - At query time, find the gallery entry with highest cosine similarity
  - If similarity > threshold, return that vehicle_id as match

This is the same paradigm as face recognition — just for vehicles.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ReIDResult:
    vehicle_id: str
    similarity: float       # cosine similarity [0, 1]
    is_new_vehicle: bool    # True if no gallery match found


@dataclass
class GalleryEntry:
    vehicle_id: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    max_embeddings: int = 10    # keep last N embeddings, average them

    def add_embedding(self, emb: np.ndarray) -> None:
        self.embeddings.append(emb)
        if len(self.embeddings) > self.max_embeddings:
            self.embeddings.pop(0)

    def get_mean_embedding(self) -> np.ndarray:
        return np.mean(self.embeddings, axis=0)


class VehicleReID:
    """
    Embedding-based vehicle re-identification.

    Uses a ResNet backbone (torchvision) fine-tuned conceptually for vehicle
    appearance. In production, you'd fine-tune on VeRi-776 or a site-specific
    dataset. Here we use ImageNet-pretrained ResNet50 as a baseline feature
    extractor — good enough for same-site matching.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity to accept a gallery match.
        0.8 = fairly strict (recommended for safety-critical matching).
    device : str
        'cpu' or 'cuda'
    """

    def __init__(self, similarity_threshold: float = 0.80, device: str = "cpu"):
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._model = None
        self._transform = None
        self._gallery: dict[str, GalleryEntry] = {}
        self._next_vehicle_id = 1

    def _load_model(self):
        """Lazy-load ResNet50 feature extractor."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove the final classification layer — use 2048-dim pool features
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            model.to(self.device)
            self._model = model

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        except ImportError:
            raise ImportError("torch and torchvision required: pip install torch torchvision")

    def extract_embedding(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """
        Extract a normalized embedding vector from a vehicle crop.

        Parameters
        ----------
        vehicle_crop : BGR image of vehicle (any size)

        Returns
        -------
        L2-normalized embedding vector of shape (2048,)
        """
        if self._model is None:
            self._load_model()

        import torch

        # BGR → RGB for torchvision
        rgb_crop = vehicle_crop[:, :, ::-1].copy()
        tensor = self._transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model(tensor)
            embedding = features.squeeze().cpu().numpy()

        # L2 normalize — cosine similarity becomes dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def query(
        self, vehicle_crop: np.ndarray
    ) -> ReIDResult:
        """
        Query the gallery with a vehicle crop.

        Parameters
        ----------
        vehicle_crop : BGR image of the vehicle to identify

        Returns
        -------
        ReIDResult with matched vehicle_id or a new ID if no match.
        """
        embedding = self.extract_embedding(vehicle_crop)
        return self.query_embedding(embedding)

    def query_embedding(self, embedding: np.ndarray) -> ReIDResult:
        """
        Query gallery with a pre-computed embedding.
        Useful when embedding was computed externally (e.g., on GPU node).
        """
        if not self._gallery:
            new_id = self._register_new(embedding)
            return ReIDResult(vehicle_id=new_id, similarity=0.0, is_new_vehicle=True)

        best_id = None
        best_sim = -1.0

        for entry in self._gallery.values():
            gallery_emb = entry.get_mean_embedding()
            # Cosine similarity: both vectors are L2-normalized, so this is dot product
            sim = float(np.dot(embedding, gallery_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = entry.vehicle_id

        if best_sim >= self.similarity_threshold:
            # Update gallery with this new embedding (online learning)
            self._gallery[best_id].add_embedding(embedding)
            return ReIDResult(vehicle_id=best_id, similarity=best_sim, is_new_vehicle=False)
        else:
            new_id = self._register_new(embedding)
            return ReIDResult(vehicle_id=new_id, similarity=best_sim, is_new_vehicle=True)

    def register_known_vehicle(self, vehicle_id: str, embedding: np.ndarray) -> None:
        """
        Pre-register a known vehicle (e.g., at parking entry with plate scan).
        Allows Re-ID to find it later without a full plate view.
        """
        if vehicle_id not in self._gallery:
            self._gallery[vehicle_id] = GalleryEntry(vehicle_id=vehicle_id)
        self._gallery[vehicle_id].add_embedding(embedding)

    def remove_vehicle(self, vehicle_id: str) -> None:
        """Remove a vehicle from the gallery when it exits the facility."""
        self._gallery.pop(vehicle_id, None)

    def gallery_size(self) -> int:
        return len(self._gallery)

    def _register_new(self, embedding: np.ndarray) -> str:
        new_id = f"VEH-{self._next_vehicle_id:05d}"
        self._next_vehicle_id += 1
        entry = GalleryEntry(vehicle_id=new_id)
        entry.add_embedding(embedding)
        self._gallery[new_id] = entry
        return new_id
