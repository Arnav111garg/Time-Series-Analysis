# Time-Series-Analysis
A practitioner-focused Python codebase accompanying [_Time Series Analysis: The Complete Practitioner's Guide for Quantitative Finance_](https://drive.google.com/drive/folders/1wGcP0FED5fF7vCwqmzfMTSj2s4ngRGzo). This repository bridges classical econometric theory with advanced quantitative finance and algorithmic trading strategies.

Core Modules & Architecture
---------------------------

*   **Foundations & Linear Models:** Classical STL decomposition, stationarity tests (ADF, KPSS, Phillips-Perron, Zivot-Andrews), ARMA/ARIMA modeling, and Box-Jenkins diagnostics.
    
*   **Volatility Dynamics:** Clustering, GARCH family models (EGARCH, GJR-GARCH), Realized Volatility (HAR-RV), and Dynamic Conditional Correlation (DCC-GARCH).
    
*   **Multivariate & State-Space:** Vector Autoregressions (VAR), Engle-Granger and Johansen cointegration, Vector Error Correction Models (VECM), Kalman Filters, and Hidden Markov Models (HMM) for regime identification.
    
*   **Advanced Quant & High-Frequency:** Nonlinear SETAR/STAR threshold models, Hawkes point processes, Amihud illiquidity, VPIN toxicity metrics, and Deep Learning (LSTMs, Temporal Fusion Transformers).
    

End-to-End Case Study
---------------------

Includes an institutional-grade Volatility Risk Premium (VRP) harvesting strategy that integrates rolling GARCH forecasts, VIX implied volatility spreads, and 3-state HMM regime filtering to dynamically size short-volatility and equity hedge positions.
