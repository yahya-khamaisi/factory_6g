"""
Receiver components for 6G smart factory physical layer.

This module implements the receiver chain for OFDM-MIMO systems, including
channel estimation, equalization, demapping, and LDPC decoding. The receiver
processes the received signal to recover the transmitted information bits.

Theory:
    The receiver chain performs the following operations:
    
    1. Channel Estimation:
       - Estimates channel frequency response H[k] from received pilots
       - LS (Least Squares) estimator: Ĥ_LS = Y_pilot / X_pilot
       - Interpolation: Estimates channel for data subcarriers from pilot estimates
       - Neural refinement: ML-based enhancement of LS estimates
       - Channel estimation error: ε = H - Ĥ, error variance σ²_ε
       
    2. Equalization:
       - Compensates for channel distortion: x̂ = f(y, Ĥ)
       - ZF (Zero-Forcing): x̂_ZF = (Ĥ^H·Ĥ)^(-1)·Ĥ^H·y
       - MMSE (Minimum Mean Square Error): x̂_MMSE = (Ĥ^H·Ĥ + σ²_n·I)^(-1)·Ĥ^H·y
       - LMMSE accounts for channel estimation error: σ²_eff = σ²_n + σ²_ε·|x|²
       
    3. Demapping:
       - Converts equalized symbols to log-likelihood ratios (LLRs)
       - LLR for bit b_i: LLR_i = log(P(b_i=1|y)/P(b_i=0|y))
       - For AWGN: LLR_i ≈ (1/σ²)·(d²_0 - d²_1), where d is distance to constellation
       - APP (A posteriori probability) demapping uses soft information
       
    4. LDPC Decoding:
       - Iterative belief propagation decoding
       - Message passing on Tanner graph
       - Check node update: L_c = 2·atanh(Π tanh(L_v/2))
       - Variable node update: L_v = L_ch + Σ L_c
       - Convergence: Decoder stops when H·ĉ^T = 0 (mod 2) or max iterations
       
    Mathematical Formulation:
        y → [Channel Estimation] → Ĥ, σ²_ε
        y, Ĥ, σ²_ε → [Equalization] → x̂, σ²_eff
        x̂, σ²_eff → [Demapping] → LLR
        LLR → [LDPC Decoder] → b̂
        
        where:
        - y: Received signal [batch, num_rx, num_streams, num_ofdm_sym, fft_size]
        - Ĥ: Channel estimate [batch, num_rx, num_tx, num_streams, num_ofdm_sym, fft_size]
        - x̂: Equalized symbols [batch, num_tx, num_streams, num_data_symbols]
        - LLR: Log-likelihood ratios [batch, num_tx, num_streams, num_coded_bits]
        - b̂: Decoded bits [batch, num_tx, num_streams, num_info_bits]

References:
    - Proakis & Salehi, "Digital Communications" (Equalization, Demapping)
    - Richardson & Urbanke, "Modern Coding Theory" (LDPC decoding)
    - Tse & Viswanath, "Fundamentals of Wireless Communication" (Channel estimation)
    - 3GPP TS 38.211: Physical channels and modulation
"""

import tensorflow as tf
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
    LSChannelEstimator,
    LMMSEEqualizer,
    RemoveNulledSubcarriers
)
from sionna.phy.mapping import Demapper
from sionna.phy.fec.ldpc import LDPC5GDecoder



class Receiver:
    """
    Receiver chain: Channel Estimation -> Equalization -> Demapping -> LDPC Decoding.
    
    This class implements the complete receiver processing chain for OFDM-MIMO
    systems. It estimates the channel, equalizes the received signal, demaps
    symbols to soft bits (LLRs), and decodes the LDPC code to recover information bits.
    
    Theory:
        The receiver performs the following signal processing steps:
        
        1. Channel Estimation:
           - Estimates channel from pilot symbols
           - Interpolates to data subcarriers
           - Provides channel estimate and error variance
           
        2. Equalization:
           - Compensates for channel distortion
           - LMMSE equalization accounts for channel estimation error
           - Outputs equalized symbols and effective noise variance
           
        3. Demapping:
           - Converts symbols to log-likelihood ratios (LLRs)
           - Uses a posteriori probability (APP) demapping
           - Provides soft information for decoding
           
        4. LDPC Decoding:
           - Iterative belief propagation decoding
           - Uses soft information (LLRs) from demapper
           - Outputs hard-decoded information bits
    """
    
    def __init__(self, config: dict, resource_grid: ResourceGrid,
                 stream_management: StreamManagement, encoder,
                 perfect_csi: bool = False,
                 channel_estimator: object | None = None):
        """
        Initialize receiver components.
        
        Creates and configures all receiver chain components including channel
        estimator, equalizer, demapper, and LDPC decoder. Supports both perfect
        and imperfect CSI scenarios.
        
        Theory:
            Receiver components:
            
            1. Channel Estimator:
               - LS estimator: Ĥ_LS[k] = Y_pilot[k] / X_pilot[k]
               - Interpolation: Nearest neighbor or linear interpolation
               - Error variance: σ²_ε ≈ σ²_n / |X_pilot|² (for LS)
               
            2. LMMSE Equalizer:
               - Optimal linear equalizer accounting for channel estimation error
               - Equalizer: W = (Ĥ^H·Ĥ + (σ²_n + σ²_ε)/σ²_x · I)^(-1)·Ĥ^H
               - Effective noise: σ²_eff = σ²_n + σ²_ε·E[|x|²]
               
            3. APP Demapper:
               - Computes LLRs using a posteriori probabilities
               - LLR_i = log(Σ_{x: b_i=1} P(x|y) / Σ_{x: b_i=0} P(x|y))
               - For AWGN: LLR_i ≈ (1/σ²_eff)·(min_{x: b_i=0} |y - Ĥ·x|² - min_{x: b_i=1} |y - Ĥ·x|²)
               
            4. LDPC Decoder:
               - Iterative belief propagation (sum-product algorithm)
               - Uses Tanner graph representation
               - Convergence when codeword satisfies parity checks
            
        Args:
            config: System configuration parameters including modulation order
                and other receiver settings.
            resource_grid: OFDM resource grid defining the time-frequency
                structure, including pilot pattern.
            stream_management: Stream management for MIMO systems, defining
                which streams belong to which transmitter-receiver pairs.
            encoder: LDPC encoder instance (needed by decoder for code structure).
            perfect_csi: If True, skip channel estimation and use perfect channel
                knowledge. If False, use channel estimator.
            channel_estimator: Optional custom channel estimator. If None and
                perfect_csi=False, uses default LS channel estimator.
        """
        super().__init__()
        self.config = config
        self.resource_grid = resource_grid
        self.stream_management = stream_management
        self.perfect_csi = perfect_csi
        
        # Initialize components
        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(resource_grid)
        
        self._default_channel_estimator = None
        if perfect_csi:
            # Perfect CSI: no channel estimation needed
            self._channel_estimator = None
        else:
            if channel_estimator is not None:
                # Use custom channel estimator (e.g., neural estimator)
                self._channel_estimator = channel_estimator
            else:
                # Default: LS channel estimator with nearest-neighbor interpolation
                self._default_channel_estimator = LSChannelEstimator(
                    resource_grid,
                    interpolation_type="nn"  # Nearest neighbor interpolation
                )
                self._channel_estimator = self._default_channel_estimator
        
        # LMMSE equalizer: optimal linear equalizer for MIMO-OFDM
        self._equalizer = LMMSEEqualizer(resource_grid, stream_management)
        
        # APP demapper: converts symbols to log-likelihood ratios
        self._demapper = Demapper("app", "qam", self.config.get("num_bits_per_symbol", 2))
        
        # LDPC decoder: use Sionna's 5G decoder
        self._decoder = LDPC5GDecoder(encoder, 
                                      num_iter=self.config.get("num_decoding_iter", 20),
                                      hard_out=True, 
                                      return_num_iter=True)
        # Run belief propagation as a traced graph. Called eagerly, the
        # decoder's while_loop over ragged tensors retains native memory on
        # every call, which exhausted a 16 GB container after ~1500 Monte Carlo
        # batches. The graph gives the same block-error decisions as eager
        # decoding; inside codewords it fails to decode, floating-point
        # ordering can flip a few hard decisions (2 of ~2e6 LS bits at 0 dB in
        # a 120-batch check), so bit-exact agreement with eager runs is not
        # guaranteed there.
        self._decode_graph = tf.function(self._decoder, reduce_retracing=True)
    
    def estimate_channel(self, y: tf.Tensor, noise_var: tf.Tensor) -> tuple:
        """
        Estimate channel from received signal.
        
        Estimates the channel frequency response using pilot symbols and
        interpolation. The channel estimate is used for equalization.
        
        Theory:
            Channel estimation process:
            
            1. Pilot-based estimation:
               - Extract received pilots: Y_pilot = H·X_pilot + N
               - LS estimate: Ĥ_pilot = Y_pilot / X_pilot
               - For MIMO: Ĥ_pilot[i,j] = Y_pilot[i] / X_pilot[j]
               
            2. Interpolation:
               - Nearest neighbor: Ĥ_data = Ĥ_pilot[nearest_pilot]
               - Linear: Ĥ_data = interpolate(Ĥ_pilot, positions)
               - More sophisticated: 2D interpolation (time-frequency)
               
            3. Error variance:
               - LS error: σ²_ε ≈ σ²_n / |X_pilot|²
               - Accounts for noise in channel estimate
               - Used by equalizer for optimal performance
            
        Args:
            y: Received signal in resource grid format
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            noise_var: Noise variance per resource element
                Used to compute channel estimation error variance
                
        Returns:
            Tuple of:
            - h_hat: Channel estimate
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
            - err_var: Channel estimation error variance
                Same shape as h_hat, or scalar if uniform error variance
                
        Raises:
            ValueError: If perfect_csi is True (should use true channel instead)
                or if channel estimator is not initialized.
        """
        if self.perfect_csi:
            # Perfect CSI case - should be handled separately with true channel
            raise ValueError("Perfect CSI should use true channel, not estimation")
        else:
            if self._channel_estimator is None:
                raise ValueError("Channel estimator is not initialized")
            return self._channel_estimator(y, noise_var)
    
    def equalize(self, y: tf.Tensor, h_hat: tf.Tensor, err_var: tf.Tensor,
                 noise_var: tf.Tensor) -> tuple:
        """
        Equalize received signal using channel estimate.
        
        Compensates for channel distortion using linear MMSE equalization.
        The equalizer accounts for both noise and channel estimation error.
        
        Theory:
            LMMSE equalization:
            
            For MIMO systems, the received signal is:
            y = H·x + n
            
            The LMMSE equalizer computes:
            x̂ = W·y
            
            where the equalizer matrix is:
            W = (Ĥ^H·Ĥ + (σ²_n + σ²_ε)/σ²_x · I)^(-1)·Ĥ^H
            
            The effective noise variance is:
            σ²_eff = σ²_n + σ²_ε·E[|x|²]
            
            This accounts for:
            - Channel noise: σ²_n
            - Channel estimation error: σ²_ε
            - Inter-stream interference (for MIMO)
            
            For perfect CSI (σ²_ε = 0):
            W = (H^H·H + σ²_n/σ²_x · I)^(-1)·H^H
            σ²_eff = σ²_n
            
        Args:
            y: Received signal
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            h_hat: Channel estimate
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
            err_var: Channel estimation error variance
                Same shape as h_hat, or scalar if uniform
            noise_var: Noise variance
                Scalar or tensor with same shape as resource grid
                
        Returns:
            Tuple of:
            - x_hat: Equalized symbols
                Shape: [batch_size, num_tx, num_streams, num_data_symbols]
            - no_eff: Effective noise variance
                Same shape as x_hat, used for demapping
        """
        return self._equalizer(y, h_hat, err_var, noise_var)
    
    def demap(self, x_hat: tf.Tensor, no_eff: tf.Tensor) -> tf.Tensor:
        """
        Demap equalized symbols to log-likelihood ratios (LLRs).
        
        Converts equalized symbols to soft bit information (LLRs) using
        a posteriori probability (APP) demapping. The LLRs are used by
        the LDPC decoder for soft-decision decoding.
        
        Theory:
            APP demapping computes log-likelihood ratios:
            
            LLR_i = log(P(b_i = 1 | x̂) / P(b_i = 0 | x̂))
            
            For AWGN channel with effective noise variance σ²_eff:
            
            LLR_i ≈ (1/σ²_eff) · (min_{x: b_i=0} |x̂ - x|² - min_{x: b_i=1} |x̂ - x|²)
            
            where the minimization is over all constellation points x with
            bit i equal to 0 or 1, respectively.
            
            Higher |LLR_i| indicates higher confidence in the bit value.
            Positive LLR_i favors b_i = 1, negative favors b_i = 0.
            
            The effective noise variance σ²_eff accounts for:
            - Channel noise
            - Channel estimation error
            - Equalization residual errors
            
        Args:
            x_hat: Equalized symbols
                Shape: [batch_size, num_tx, num_streams, num_data_symbols]
            no_eff: Effective noise variance
                Same shape as x_hat
                Used to compute LLR scaling: LLR ∝ 1/σ²_eff
                
        Returns:
            Log-likelihood ratios (LLRs)
            Shape: [batch_size, num_tx, num_streams, num_coded_bits]
            Higher values indicate higher confidence in bit = 1
        """
        llr = self._demapper(x_hat, no_eff)
        # Clip extreme LLRs to prevent numerical issues and improve decoder performance
        # Diagnostic showed 27.5% of LLRs have |LLR| > 50, which can cause decoder issues
        # Clip to reasonable range: ±20 is typical for LDPC decoders
        llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
        return llr_clipped
    
    def decode(self, llr: tf.Tensor) -> tuple:
        """
        Decode LLRs to information bits using LDPC decoder.
        
        Performs iterative belief propagation decoding on the LDPC code
        to recover the transmitted information bits from the soft bit
        information (LLRs).
        
        Theory:
            LDPC decoding uses iterative belief propagation:
            
            1. Initialization:
               - Variable nodes: L_v = LLR (channel LLRs)
               
            2. Check node update:
               - For each check node, update messages to variable nodes:
               L_c = 2·atanh(Π_{v'≠v} tanh(L_v' / 2))
               
            3. Variable node update:
               - For each variable node, update messages to check nodes:
               L_v = L_ch + Σ_{c'≠c} L_c'
               
            4. Decision:
               - Hard decision: b̂_i = 1 if L_v > 0, else 0
               - Check parity: H·ĉ^T = 0 (mod 2)?
               - If yes or max iterations: stop, else continue
            
            The decoder converges when:
            - All parity checks are satisfied: H·ĉ^T = 0
            - Maximum number of iterations reached
            - Early stopping criteria met
            
        Args:
            llr: Log-likelihood ratios from demapper
                Shape: [batch_size, num_tx, num_streams, num_coded_bits]
                
        Returns:
            Decoded information bits
            Shape: [batch_size, num_tx, num_streams, num_info_bits]
            Values are 0 or 1 (hard decisions)
        """
        decoder_out = self._decode_graph(llr)
        if isinstance(decoder_out, tuple):
            if len(decoder_out) == 3:
                decoded, _, num_iter = decoder_out
            elif len(decoder_out) == 2:
                decoded, num_iter = decoder_out
            else:
                decoded = decoder_out[0]
                num_iter = decoder_out[-1]
        else:
            decoded = decoder_out
            num_iter = tf.zeros_like(decoded[..., 0], dtype=tf.float32)
        num_iter = tf.cast(num_iter, tf.float32)
        return decoded, num_iter
    
    def call(self, y: tf.Tensor, h_hat: tf.Tensor, err_var: tf.Tensor,
             noise_var: tf.Tensor) -> tf.Tensor:
        """
        Complete receiver chain processing.
        
        Processes the received signal through the complete receiver chain:
        1. Equalize received signal
        2. Demap symbols to LLRs
        3. Decode LLRs to information bits
        
        This method assumes channel estimation has already been performed.
        For perfect CSI, use process_with_perfect_csi() instead.
        
        Theory:
            The complete receiver processing:
            
            y, Ĥ, σ²_ε → [Equalization] → x̂, σ²_eff
            x̂, σ²_eff → [Demapping] → LLR
            LLR → [LDPC Decoder] → b̂
            
            The performance depends on:
            - Channel estimation accuracy (σ²_ε)
            - Equalization method (LMMSE is optimal)
            - Demapping quality (APP is optimal)
            - Decoding iterations (more iterations = better performance)
            
        Args:
            y: Received signal
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            h_hat: Channel estimate (or true channel for perfect CSI)
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
            err_var: Channel estimation error variance
                For perfect CSI, set to 0.0
                Shape: same as h_hat, or scalar
            noise_var: Noise variance
                Scalar or tensor with same shape as resource grid
                
        Returns:
            Decoded information bits
            Shape: [batch_size, num_tx, num_streams, num_info_bits]
        """
        # Equalize
        x_hat, no_eff = self.equalize(y, h_hat, err_var, noise_var)
        
        # Demap to LLRs
        llr = self.demap(x_hat, no_eff)
        
        # Decode
        b_hat, _ = self.decode(llr)
        
        return b_hat
    
    def __call__(self, y: tf.Tensor, h_hat: tf.Tensor, err_var: tf.Tensor,
                 noise_var: tf.Tensor) -> tf.Tensor:
        """
        Alias for call method for convenience.
        
        Allows the receiver to be called as a function:
        b_hat = receiver(y, h_hat, err_var, noise_var)
        
        Args:
            y: Received signal
            h_hat: Channel estimate
            err_var: Channel estimation error variance
            noise_var: Noise variance
            
        Returns:
            Decoded information bits
        """
        return self.call(y, h_hat, err_var, noise_var)
    
    def process_with_perfect_csi(self, y: tf.Tensor, h: tf.Tensor,
                                  noise_var: tf.Tensor) -> tf.Tensor:
        """
        Process received signal with perfect channel knowledge.
        
        Processes the received signal assuming perfect channel state information
        (no channel estimation error). This is used for performance upper bound
        analysis and comparison with imperfect CSI scenarios.
        
        Theory:
            With perfect CSI (H known exactly, σ²_ε = 0):
            
            - Equalization: Optimal LMMSE with no estimation error
            - Effective noise: σ²_eff = σ²_n (no estimation error contribution)
            - Performance: Upper bound on achievable performance
            - Comparison: Used to measure channel estimation penalty
            
            The performance gap between perfect and imperfect CSI indicates:
            - Channel estimation quality
            - Pilot overhead impact
            - Potential for improved estimators
            
        Args:
            y: Received signal
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            h: True channel response (perfect CSI)
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
            noise_var: Noise variance
                Scalar or tensor with same shape as resource grid
                
        Returns:
            Decoded information bits
            Shape: [batch_size, num_tx, num_streams, num_info_bits]
        """
        # Remove nulled subcarriers from both received signal and channel
        # (match resource grid structure)
        # y_processed = self._remove_nulled_subcarriers(y)
        # h_hat = self._remove_nulled_subcarriers(h)
        
        # Perfect CSI: no channel estimation error
        err_var = 0.0
        
        # Process through receiver chain
        # Pass full y and h (h_hat) to receiver, as equalizer expects full grid
        return self.__call__(y, h, err_var, noise_var)
