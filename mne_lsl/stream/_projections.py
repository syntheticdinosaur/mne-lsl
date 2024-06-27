from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

TYPE_CHECKING = True

import numpy as np
from scipy.linalg import eigh, pinv

import mne

if TYPE_CHECKING:
    from typing import Any, Optional

    
class StreamProjection():
    """ Class definition an individual projection."""
    
    def __init__(self):
        self._matrix = None
        self._desc   = None
        self._kind   = None
        self._source = None
        self._rank   = None
        self._active = True
        
    @property
    def active(self):
        return self._active
        
    @property
    def matrix(self):
        return self._matrix
    
    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix
        
    @matrix.getter
    def matrix(self):
        return self._matrix
    
        
    def __repr__(self) -> str:
        return f"<Projection: {self._desc} ({self._kind})>"
    
    def __eq__(self, other: Any) -> bool:
        return self._proj == other._proj and self._desc == other._desc and self._kind == other._kind 
    
    def from_ica(self, ica, components:list=None):
        """Based on mne ica._pick_sources()"""
        if n_pca_components is None:
            n_pca_components = ica.n_pca_components
            
        include = components
        exclude = ica._check_exclude(exclude) #!
        _n_pca_comp = ica._check_n_pca_components(n_pca_components)
        n_ch, _ = data.shape

        max_pca_components = ica.pca_components_.shape[0]
        if not ica.n_components_ <= _n_pca_comp <= max_pca_components:
            raise ValueError(
                f"n_pca_components ({_n_pca_comp}) must be >= "
                f"n_components_ ({ica.n_components_}) and <= "
                "the total number of PCA components "
                f"({max_pca_components})."
            )

        # Apply first PCA
        if ica.pca_mean_ is not None:
            data -= ica.pca_mean_[:, None]

        sel_keep = np.arange(ica.n_components_)
        if include not in (None, []):
            sel_keep = np.unique(include)
        elif exclude not in (None, []):
            sel_keep = np.setdiff1d(np.arange(ica.n_components_), exclude)

        n_zero = ica.n_components_ - len(sel_keep)

        # Mixing and unmixing should both be shape (self.n_components_, 2),
        # and we need to put these into the upper left part of larger mixing
        # and unmixing matrices of shape (n_ch, _n_pca_comp)
        pca_components = ica.pca_components_[:_n_pca_comp]
        assert pca_components.shape == (_n_pca_comp, n_ch)
        assert (
            ica.unmixing_matrix_.shape
            == ica.mixing_matrix_.shape
            == (ica.n_components_,) * 2
        )
        unmixing = np.eye(_n_pca_comp)
        unmixing[: ica.n_components_, : ica.n_components_] = ica.unmixing_matrix_
        unmixing = np.dot(unmixing, pca_components)

        mixing = np.eye(_n_pca_comp)
        mixing[: ica.n_components_, : ica.n_components_] = ica.mixing_matrix_
        mixing = pca_components.T @ mixing
        assert mixing.shape == unmixing.shape[::-1] == (n_ch, _n_pca_comp)

        # keep requested components plus residuals (if any)
        sel_keep = np.concatenate(
            (sel_keep, np.arange(ica.n_components_, _n_pca_comp))
        )
       
        self.matrix = np.dot(mixing[:, sel_keep], unmixing[sel_keep, :])
        self.desc   = f"ICA components {sel_keep}"
        self.kind   = "ICA"
        self.source = "ICA"
    
        """        data = np.dot(proj_mat, data)
                assert proj_mat.shape == (n_ch,) * 2

                if ica.pca_mean_ is not None:
                    data += ica.pca_mean_[:, None]

                # restore scaling
                if ica.noise_cov is None:  # revert standardization
                    data *= ica.pre_whitener_
                else:
                    data = np.linalg.pinv(ica.pre_whitener_, rcond=1e-14) @ data"""
            
    def from_ssp(self, ssp, components:list=None):
        """ SSP projection matrix is defined as I - U*U^T, where U is the orthogonalized basis of the subspace.
            MNE only passes the vectors U, so we need to compute the projection matrix. """
        # For now, assume single SSP, but expand to list later.
        # What to do with bad channels?
        U = ssp["data"]["data"].T
        I = np.eye(U.shape[0])
        
        self.matrix  = I - U@U.T
        self._desc   = ssp["desc"]
        self._kind   = ssp["kind"]
        self._source = "SSP"
    
    def from_csp(self, csp, components:list=None):        
        # Sorting eigenvectors by descending order of eigenvalues
        # But CSP doesnt return eigenvalues, which is sad.
        suppression_matrix = csp.filters_.copy()
        suppression_matrix[:, :n_components] = 0
        
        matrix  = csp.filters_[:, :n_components] @ csp.patterns_[:n_components, :] # or csp.patterns_
        # Applying GED for artifact suppression
        self.matrix = matrix
        self.desc   = csp["desc"]
        self.kind   = csp["kind"]
        self.source = "CSP/GEVD"
        
    def from_cov(self,
                 covArtifact:np.ndarray,
                 covRef:np.ndarray = None,
                 n_components:int= None,
                 regularize:float = 1e-10):
        """ Uses eigendecomposition to compute the projection matrix.
            Uses by default regularization, set regularize to 0 to not regularize.
            If covB is None, it computes the eigendecomposition of covA (~PCA).
            If CovB is given, it computes the generalized eigendecomposition of covA and covB (~LDA)."""

        assert covArtifact.shape[0] == covArtifact.shape[1], "Covariance matrix must be square."
        assert covArtifact.shape[0 > n_components], "Number of components must be less than number of channels."
        assert regularize >= 0, "Regularization must be positive."
        
        
        reg = np.eye(covArtifact.shape[0]) * regularize
        if covRef is None:
            evals, evecs = eigh(covArtifact + reg)
        else:
            evals, evecs = eigh(covArtifact + reg,
                                covRef + reg)  
                  
        sidx  = np.argsort(evals)[::-1]
        evecs = evecs[:, sidx]
        inv   = pinv(evecs)
        
        evecs = evecs.T
        evecs[:, :n_components] = 0
        
        self.matrix = np.dot(evecs, inv) #np.dot(evecs[:,n_components:], inv[n_components:,:])
        self._desc   = f"Eigenvalues {evals[sidx]}"
        self._kind   = "Covariance"
        self._source = "Eigendecomposition"
       
    
    def from_custom(self,
                    matrix:np.ndarray,
                    desc:str,
                    kind:str):
        """ Matrix is a 2D numpy array, desc is a string, and kind is a string.
            Matrix must be of shape (n_channels, n_channels). """
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
        self.matrix = matrix
        self.desc   = desc
        self.kind   = kind
        
    def apply_array(self, data:np.ndarray):
        assert data.shape[0] == self.matrix.shape[0], f"Data {data.shape} and matrix {self.matrix.shape} must have same number of channels."
        return np.dot(self.matrix, data)        
    
    def apply_raw(self,
                  raw:mne.io.RawArray,
                  copy=True):   
        data = raw.get_data() # picks?   
        assert data.shape[0] == self._matrix.shape[0], "Data and matrix must have same number of channels."
        filtered = np.dot(self._matrix, data)        
        if copy:
            raw = raw.copy()
            raw._data = filtered  
        else:
            raw._data = filtered
        return raw
        
    def apply_epochs(self, epochs, copy=True):
        pass
    
    def apply_evoked(self, evoked, copy=True):
        pass
    

covA = np.random.randn(10)
covA = covA[None,:].T@covA[None,:] + np.eye(10)*0.1

covB = np.random.randn(10)
covB = covB[None,:].T@covB[None,:] + np.eye(10)*0.1



streamProj = StreamProjection()
streamProj.from_cov(covA, covB, 1)
