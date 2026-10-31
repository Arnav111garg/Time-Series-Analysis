# Time-Series-Analysis
A practitioner-focused Python codebase designed for quantitative finance and algorithmic trading. This repository bridges classical econometric theory with advanced quantitative modeling, deep learning architectures, and systematic strategy backtesting.

Repository Architecture & Core Modules
--------------------------------------

*   **Foundations & Stationarity:** Classical STL decomposition, log-return transformations, and comprehensive unit root testing including ADF, Phillips–Perron, KPSS, and Zivot–Andrews tests to model structural breaks.
    
*   **Linear Time Series & Diagnostics:** Complete Box–Jenkins methodology (identification via ACF/PACF/EACF, information criteria AIC/BIC/HQC, Ljung–Box and BDS white noise tests) covering AR, MA, ARMA, ARIMA, and Seasonal ARIMA (SARIMA) formulations.
    
*   **Volatility Dynamics & High-Frequency:** Stylized return facts, asymmetric GARCH models (GJR-GARCH, EGARCH, APARCH), dynamic conditional correlations (DCC-GARCH), realized volatility modeling via HAR-RV, Roll bid-ask spread estimators, intraday U-shape volatility patterns, and self-exciting Hawkes point processes for trade arrival clustering.
    
*   **Multivariate & State-Space Systems:** Structural Vector Autoregressions (VAR), Engle–Granger two-step and Johansen cointegration tests, Vector Error Correction Models (VECM), Toda–Yamamoto Granger causality, dynamic Kalman filtering for time-varying betas and hedge ratios, and Hidden Markov Models (HMM) for regime classification via Baum–Welch and Viterbi algorithms.
    
*   **Nonlinear & Advanced Quant Models:** Threshold models including SETAR and ESTAR/LSTAR smooth transition dynamics for transaction-cost bands, Copulas (Gaussian, Student-t, Archimedean, Vine) for tail-dependence modeling, Extreme Value Theory (GEV, POT/GPD), ARFIMA fractional integration, dynamic factor models, and Bayesian structural time series (BSTS).
    
*   **Machine Learning & Deep Learning:** Feature engineering pipelines, gradient boosting return forecasters, LSTMs, Temporal Fusion Transformers (TFT), and Reinforcement Learning (DQN, PPO) tailored for automated trading.
    

Comprehensive Strategy Implementation
-------------------------------------

The codebase features an institutional-grade Volatility Risk Premium (VRP) harvesting backtest (Module 18). The strategy fits rolling GARCH(1,1) models on SPY to forecast 1-day ahead conditional realized volatility, measures the spread against VIX implied volatility, and filters execution through a 3-state Gaussian HMM. Positions in short-volatility assets (VXX) and dynamic SPY equity hedges are sized dynamically according to posterior regime probabilities and z-score thresholds, taking into account realistic 10 bps transaction costs.
