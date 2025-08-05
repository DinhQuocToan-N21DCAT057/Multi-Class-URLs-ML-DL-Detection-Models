# Overview

This is a Flask-based web application for multi-label URL malicious detection using deep learning models. The system analyzes URLs to classify them into multiple categories: benign, defacement, malware, and phishing. It provides a comprehensive web interface for URL security scanning with support for multiple machine learning models (CNN, XGBoost, Random Forest) and datasets.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses a modern web frontend built with:
- **Template Engine**: Jinja2 templates with Bootstrap 5 for responsive UI
- **Styling**: Glass morphism design with CSS custom properties and gradient backgrounds
- **JavaScript**: Vanilla JS with Chart.js for data visualization and dynamic interactions
- **Components**: Modular template structure with base templates and component includes

## Backend Architecture
- **Framework**: Flask web framework with Python
- **Application Structure**: Modular design with separate concerns for prediction, feature extraction, and web routing
- **Model Management**: Support for multiple ML model types (CNN, XGBoost, Random Forest) with configurable datasets
- **Feature Engineering**: Comprehensive URL feature extraction including domain analysis, path analysis, and security checks

## Data Processing Pipeline
- **URL Feature Extractor**: Extracts numerical and non-numerical features from URLs using various techniques including WHOIS lookup, DNS resolution, and content analysis
- **Model Predictor**: Unified interface for running predictions across different model types with configurable thresholds
- **Data Preprocessing**: Includes dataset merging, balancing, and splitting utilities for model training

## Authentication and Session Management
- **Session Security**: Flask sessions with configurable secret keys
- **Security Middleware**: ProxyFix middleware for deployment behind reverse proxies

## Configuration Management
- **Environment-based Config**: Centralized configuration using environment variables
- **Model Paths**: Configurable model file locations with support for multiple datasets
- **Application Settings**: Configurable thresholds, file upload limits, and feature toggles

# External Dependencies

## Database and Storage
- **Firebase Realtime Database**: Used for storing prediction results and historical data
- **Firebase Admin SDK**: Server-side Firebase integration for data persistence

## Machine Learning Stack
- **TensorFlow/Keras**: For CNN model loading and inference
- **scikit-learn**: For XGBoost and Random Forest models, plus preprocessing utilities
- **NLTK**: Natural language processing for text feature extraction
- **NumPy/Pandas**: Data manipulation and numerical computing

## Web and Network Analysis
- **Requests**: HTTP client for URL content analysis
- **BeautifulSoup/PyQuery**: HTML parsing and content extraction
- **python-whois**: Domain registration information lookup
- **dnspython**: DNS resolution and analysis
- **tldextract**: Top-level domain parsing

## Infrastructure and Utilities
- **Werkzeug**: WSGI utilities and middleware
- **Levenshtein**: String similarity calculations for URL analysis
- **concurrent.futures**: Parallel processing for feature extraction
- **hashlib**: URL hashing for caching and deduplication

## Frontend Dependencies
- **Bootstrap 5**: Responsive CSS framework
- **Font Awesome**: Icon library
- **Chart.js**: Data visualization and charting
- **CDN-hosted libraries**: External resources for UI components