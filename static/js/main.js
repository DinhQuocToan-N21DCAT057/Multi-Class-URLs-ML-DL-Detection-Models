// Main JavaScript functionality for URL Multi-Labels Detection
class URLDetectionApp {
    constructor() {
        this.settings = this.loadSettings();
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeComponents();
        this.loadInitialData();
    }

    bindEvents() {
        // Global event listeners
        document.addEventListener('DOMContentLoaded', () => {
            this.onDOMContentLoaded();
        });

        // Form submissions
        this.bindFormEvents();

        // Navigation events
        this.bindNavigationEvents();

        // Utility events
        this.bindUtilityEvents();
    }

    onDOMContentLoaded() {
        // Initialize tooltips
        this.initializeTooltips();

        // Initialize animations
        this.initializeAnimations();

        // Load user preferences
        this.applyUserPreferences();

        // Auto-refresh data
        this.setupAutoRefresh();
    }

    bindFormEvents() {
        // Quick prediction form
        const quickForm = document.getElementById('quickPredictionForm');
        if (quickForm) {
            quickForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleQuickPrediction(e.target);
            });
        }

        // Multi-model form
        const multiForm = document.getElementById('multiModelForm');
        if (multiForm) {
            multiForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleMultiModelPrediction(e.target);
            });
        }

        // Settings form
        const settingsForm = document.getElementById('settingsForm');
        if (settingsForm) {
            settingsForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSettingsUpdate(e.target);
            });
        }

        // URL input validation
        const urlInputs = document.querySelectorAll('input[type="url"]');
        urlInputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateURL(input);
            });
        });

        // Range sliders
        const rangeInputs = document.querySelectorAll('input[type="range"]');
        rangeInputs.forEach(range => {
            range.addEventListener('input', () => {
                this.updateRangeValue(range);
            });
        });
    }

    bindNavigationEvents() {
        // Smooth scrolling for anchor links
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        anchorLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.smoothScrollTo(link.getAttribute('href'));
            });
        });

        // Active navigation highlighting
        this.updateActiveNavigation();
        window.addEventListener('scroll', () => {
            this.updateActiveNavigation();
        });
    }

    bindUtilityEvents() {
        // Copy to clipboard functionality
        const copyButtons = document.querySelectorAll('[data-copy]');
        copyButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.copyToClipboard(button.dataset.copy);
            });
        });

        // Download functionality
        const downloadButtons = document.querySelectorAll('[data-download]');
        downloadButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.downloadData(button.dataset.download, button.dataset.filename);
            });
        });

        // Auto-save forms
        const autoSaveForms = document.querySelectorAll('[data-autosave]');
        autoSaveForms.forEach(form => {
            const inputs = form.querySelectorAll('input, select, textarea');
            inputs.forEach(input => {
                input.addEventListener('change', () => {
                    this.autoSaveForm(form);
                });
            });
        });
    }

    // Prediction Methods
    async handleQuickPrediction(form) {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        this.showLoading('loadingSection');
        this.hideElement('resultsSection');

        try {
            const response = await this.makeAPICall('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: data.url,
                    model_type: data.model || this.settings.defaultModel,
                    threshold: parseFloat(data.threshold) / 100 || 0.5,
                    dataset: data.dataset || this.settings.defaultDataset,
                    numerical: true
                })
            });

            if (response.error) {
                this.showAlert(response.error, 'danger');
            } else {
                this.displayPredictionResults(response);
            }
        } catch (error) {
            this.showAlert('Có lỗi xảy ra khi phân tích URL!', 'danger');
            console.error('Prediction error:', error);
        } finally {
            this.hideLoading('loadingSection');
        }
    }

    async handleMultiModelPrediction(form) {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        this.showLoading('loadingSection');
        this.hideElement('resultsSection');
        this.animateProgress('loadingProgress');

        try {
            const response = await this.makeAPICall('/predict-multi', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: data.url,
                    threshold: parseFloat(data.threshold) / 100 || 0.5,
                    dataset: data.dataset || this.settings.defaultDataset,
                    numerical: true
                })
            });

            if (response.error) {
                this.showAlert(response.error, 'danger');
            } else {
                this.displayMultiModelResults(response);
            }
        } catch (error) {
            this.showAlert('Có lỗi xảy ra khi phân tích URL!', 'danger');
            console.error('Multi-model prediction error:', error);
        } finally {
            this.hideLoading('loadingSection');
        }
    }

    displayPredictionResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const overallAssessment = document.getElementById('overallAssessment');
        const predictionBars = document.getElementById('predictionBars');

        // Calculate safety assessment
        const assessment = this.calculateSafetyAssessment(data.predictions, data.threshold);

        // Update overall assessment
        overallAssessment.className = `alert ${assessment.is_safe ? 'alert-success' : 'alert-danger'}`;
        overallAssessment.innerHTML = this.generateAssessmentHTML(assessment, data.url);

        // Create prediction bars
        if (predictionBars) {
            predictionBars.innerHTML = this.generatePredictionBarsHTML(data.predictions, data.label_names, data.threshold);
        }

        // Update execution info
        this.updateExecutionInfo(data);

        // Show results with animation
        this.showElement('resultsSection', 'fade-in');

        // Save to recent predictions
        this.saveRecentPrediction(data);
    }

    displayMultiModelResults(data) {
        const resultsSection = document.getElementById('resultsSection');

        // Update individual model cards
        const modelTypes = ['cnn', 'xgb', 'rf'];
        const modelNames = {'cnn': 'CNN', 'xgb': 'XGBoost', 'rf': 'Random Forest'};

        let safeModels = 0;
        let bestModel = null;
        let bestConfidence = 0;

        modelTypes.forEach(modelType => {
            const resultCard = document.getElementById(`${modelType}Results`);
            if (!resultCard) return;

            const modelContent = resultCard.querySelector('.model-content');
            const modelResult = data.results[modelType];

            if (modelResult.error) {
                modelContent.innerHTML = this.generateErrorHTML(modelResult.error);
                return;
            }

            const predictions = modelResult.predictions;
            const isSafe = this.isSafePrediction(predictions, data.threshold);

            if (isSafe) safeModels++;

            if (predictions[0] > bestConfidence) {
                bestConfidence = predictions[0];
                bestModel = modelType;
            }

            // Update card appearance
            resultCard.className = `model-result-card ${isSafe ? 'safe' : 'danger'}`;

            // Generate content
            modelContent.innerHTML = this.generateModelResultHTML(
                predictions, data.label_names, data.threshold, modelResult.execution_time, isSafe
            );
        });

        // Generate summary
        this.generateSummary(data, safeModels, bestModel, modelNames);

        // Create comparison chart if function exists
        if (typeof createComparisonChart === 'function') {
            createComparisonChart(data, data.label_names, modelTypes, {
                'cnn': '#667eea',
                'xgb': '#764ba2',
                'rf': '#f093fb'
            });
        }

        // Show results
        this.showElement('resultsSection', 'slide-up');
    }

    // Utility Methods
    calculateSafetyAssessment(predictions, threshold = 0.5) {
        const benignProb = predictions[0] || 0;
        const maliciousProbs = predictions.slice(1);
        const maxMaliciousProb = Math.max(...maliciousProbs);
        const isSafe = benignProb > threshold && maxMaliciousProb <= threshold;

        let riskLevel, riskColor, riskIcon;

        if (isSafe) {
            riskLevel = "An toàn";
            riskColor = "#28a745";
            riskIcon = "fa-shield-alt";
        } else if (maxMaliciousProb > 0.8) {
            riskLevel = "Rất nguy hiểm";
            riskColor = "#dc3545";
            riskIcon = "fa-exclamation-triangle";
        } else if (maxMaliciousProb > 0.6) {
            riskLevel = "Nguy hiểm";
            riskColor = "#fd7e14";
            riskIcon = "fa-exclamation-circle";
        } else {
            riskLevel = "Đáng ngờ";
            riskColor = "#ffc107";
            riskIcon = "fa-question-circle";
        }

        return {
            is_safe: isSafe,
            risk_level: riskLevel,
            risk_color: riskColor,
            risk_icon: riskIcon,
            benign_confidence: benignProb,
            max_malicious_confidence: maxMaliciousProb
        };
    }

    isSafePrediction(predictions, threshold = 0.5) {
        return predictions[0] > threshold && Math.max(...predictions.slice(1)) <= threshold;
    }

    generateAssessmentHTML(assessment, url) {
        return `
            <div class="d-flex align-items-center">
                <i class="fas ${assessment.risk_icon} me-3" style="font-size: 2rem; color: ${assessment.risk_color};"></i>
                <div>
                    <h5 class="mb-1">${assessment.risk_level}</h5>
                    <p class="mb-0">
                        ${assessment.is_safe ?
                            'URL này có khả năng an toàn cao và không chứa mối đe dọa.' :
                            'URL này có thể chứa mối đe dọa bảo mật, hãy thận trọng khi truy cập.'
                        }
                    </p>
                    <small class="text-muted">URL: ${this.truncateText(url, 60)}</small>
                </div>
            </div>
        `;
    }

    generatePredictionBarsHTML(predictions, labels, threshold) {
        const colors = ['#28a745', '#ffc107', '#dc3545', '#fd7e14'];
        let html = '';

        predictions.forEach((prob, index) => {
            const percentage = (prob * 100).toFixed(1);
            const isPositive = prob > threshold;

            html += `
                <div class="prediction-bar-item mb-3">
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <span class="fw-semibold">${labels[index]}</span>
                        <span class="badge ${isPositive ? 'bg-danger' : 'bg-secondary'}">${percentage}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar"
                             style="width: ${percentage}%; background-color: ${colors[index]};"
                             role="progressbar"></div>
                    </div>
                </div>
            `;
        });

        return html;
    }

    generateModelResultHTML(predictions, labels, threshold, executionTime, isSafe) {
        const colors = ['#667eea', '#764ba2', '#f093fb'];

        let predictionBars = '';
        predictions.forEach((prob, index) => {
            const percentage = (prob * 100).toFixed(1);

            predictionBars += `
                <div class="mini-prediction-bar mb-2">
                    <div class="d-flex justify-content-between">
                        <small>${labels[index]}</small>
                        <small class="fw-bold">${percentage}%</small>
                    </div>
                    <div class="progress" style="height: 4px;">
                        <div class="progress-bar"
                             style="width: ${percentage}%; background-color: ${colors[0]};"
                             role="progressbar"></div>
                    </div>
                </div>
            `;
        });

        return `
            <div class="safety-indicator mb-3">
                <i class="fas ${isSafe ? 'fa-shield-alt text-success' : 'fa-exclamation-triangle text-danger'}"></i>
                <span class="ms-2 fw-semibold">${isSafe ? 'An toàn' : 'Nguy hiểm'}</span>
            </div>
            ${predictionBars}
            <div class="execution-time">
                <small class="text-muted">
                    <i class="fas fa-clock me-1"></i>
                    ${(executionTime || 0).toFixed(2)}s
                </small>
            </div>
        `;
    }

    generateErrorHTML(error) {
        return `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Lỗi: ${error}
            </div>
        `;
    }

    updateExecutionInfo(data) {
        const usedModel = document.getElementById('usedModel');
        const executionTime = document.getElementById('executionTime');
        const usedDataset = document.getElementById('usedDataset');

        if (usedModel) usedModel.textContent = (data.model_type || '').toUpperCase();
        if (executionTime) executionTime.textContent = (data.execution_time || 0).toFixed(2) + 's';
        if (usedDataset) usedDataset.textContent = data.dataset || '';
    }

    generateSummary(data, safeModels, bestModel, modelNames) {
        const summaryContent = document.getElementById('summaryContent');
        if (!summaryContent) return;

        const totalModels = 3;
        const consensus = safeModels / totalModels;

        let consensusText, consensusClass, recommendation;

        if (consensus >= 0.67) {
            consensusText = 'Đồng thuận cao';
            consensusClass = 'text-success';
            recommendation = safeModels === totalModels ?
                'Tất cả các mô hình đều đánh giá URL này là an toàn.' :
                'Đa số mô hình đánh giá URL này là an toàn.';
        } else if (consensus >= 0.33) {
            consensusText = 'Kết quả hỗn hợp';
            consensusClass = 'text-warning';
            recommendation = 'Các mô hình có ý kiến khác nhau về URL này. Nên thận trọng khi truy cập.';
        } else {
            consensusText = 'Đồng thuận thấp';
            consensusClass = 'text-danger';
            recommendation = 'Đa số mô hình đánh giá URL này có nguy cơ bảo mật. Không nên truy cập.';
        }

        summaryContent.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="summary-item">
                        <h6>Mức độ đồng thuận</h6>
                        <p class="${consensusClass} fw-semibold">${consensusText} (${safeModels}/${totalModels} mô hình)</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="summary-item">
                        <h6>Mô hình tốt nhất</h6>
                        <p class="fw-semibold">${modelNames[bestModel]} (độ tin cậy cao nhất)</p>
                    </div>
                </div>
            </div>
            <div class="recommendation">
                <h6>Khuyến nghị</h6>
                <p>${recommendation}</p>
            </div>
        `;
    }

    // API Methods
    async makeAPICall(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }

    async loadSystemStats() {
        try {
            const stats = await this.makeAPICall('/api/stats');
            this.updateStatsDisplay(stats);
            return stats;
        } catch (error) {
            console.error('Failed to load system stats:', error);
            return null;
        }
    }

    async loadPredictionHistory(limit = 50) {
        try {
            const history = await this.makeAPICall(`/api/history?limit=${limit}`);
            return history;
        } catch (error) {
            console.error('Failed to load prediction history:', error);
            return [];
        }
    }

    updateStatsDisplay(stats) {
        const elements = {
            'safeCount': stats.safe_count || 0,
            'maliciousCount': stats.malicious_count || 0,
            'totalCount': stats.total_predictions || 0
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                this.animateCounter(element, value);
            }
        });
    }

    // UI Helper Methods
    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
            element.classList.add('fade-in');
        }
    }

    hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
            element.classList.remove('fade-in');
        }
    }

    showElement(elementId, animationClass = '') {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
            if (animationClass) {
                element.classList.add(animationClass);
            }
        }
    }

    hideElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    }

    showAlert(message, type = 'info', duration = 5000) {
        const alertContainer = document.querySelector('.alert-container') || document.body;
        const alertId = 'alert-' + Date.now();

        const alertHTML = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
                <i class="fas ${this.getAlertIcon(type)} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        alertContainer.insertAdjacentHTML('afterbegin', alertHTML);

        // Auto-dismiss after duration
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                alertElement.remove();
            }
        }, duration);
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'fa-check-circle',
            'danger': 'fa-exclamation-triangle',
            'warning': 'fa-exclamation-circle',
            'info': 'fa-info-circle'
        };
        return icons[type] || icons['info'];
    }

    animateCounter(element, targetValue, duration = 1000) {
        const startValue = parseInt(element.textContent) || 0;
        const increment = (targetValue - startValue) / (duration / 16);
        let currentValue = startValue;

        const timer = setInterval(() => {
            currentValue += increment;
            if ((increment > 0 && currentValue >= targetValue) ||
                (increment < 0 && currentValue <= targetValue)) {
                currentValue = targetValue;
                clearInterval(timer);
            }
            element.textContent = Math.floor(currentValue);
        }, 16);
    }

    animateProgress(elementId, duration = 3000) {
        const element = document.getElementById(elementId);
        if (!element) return;

        let progress = 0;
        const increment = 100 / (duration / 100);

        const timer = setInterval(() => {
            progress += increment;
            if (progress >= 100) {
                progress = 100;
                clearInterval(timer);
            }
            element.style.width = progress + '%';
        }, 100);
    }

    validateURL(input) {
        const url = input.value.trim();
        if (!url) return;

        const urlPattern = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;
        const isValid = urlPattern.test(url);

        if (isValid) {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        } else {
            input.classList.remove('is-valid');
            input.classList.add('is-invalid');
        }

        return isValid;
    }

    updateRangeValue(rangeElement) {
        const targetId = rangeElement.getAttribute('data-target');
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
            targetElement.textContent = rangeElement.value + '%';
        }
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showAlert('Đã sao chép vào clipboard!', 'success', 2000);
        }).catch(() => {
            this.showAlert('Không thể sao chép!', 'danger', 2000);
        });
    }

    downloadData(data, filename) {
        const blob = new Blob([data], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Settings and Preferences
    loadSettings() {
        const defaultSettings = {
            defaultModel: 'cnn',
            defaultDataset: 'dataset_1',
            thresholds: {
                cnn: 50,
                xgb: 50,
                rf: 50
            },
            enabledModels: {
                cnn: true,
                xgb: true,
                rf: true
            },
            autoRefresh: true,
            refreshInterval: 30000,
            theme: 'light'
        };

        try {
            const stored = localStorage.getItem('urlDetectionSettings');
            return stored ? { ...defaultSettings, ...JSON.parse(stored) } : defaultSettings;
        } catch (error) {
            console.error('Failed to load settings:', error);
            return defaultSettings;
        }
    }

    saveSettings(settings) {
        try {
            this.settings = { ...this.settings, ...settings };
            localStorage.setItem('urlDetectionSettings', JSON.stringify(this.settings));
            return true;
        } catch (error) {
            console.error('Failed to save settings:', error);
            return false;
        }
    }

    applyUserPreferences() {
        // Apply theme
        if (this.settings.theme === 'dark') {
            document.body.classList.add('dark-theme');
        }

        // Apply auto-refresh settings
        if (this.settings.autoRefresh) {
            this.setupAutoRefresh();
        }
    }

    setupAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        if (this.settings.autoRefresh && this.settings.refreshInterval > 0) {
            this.refreshTimer = setInterval(() => {
                this.loadSystemStats();
            }, this.settings.refreshInterval);
        }
    }

    saveRecentPrediction(data) {
        try {
            let recent = JSON.parse(localStorage.getItem('recentPredictions') || '[]');
            recent.unshift({
                url: data.url,
                model: data.model_type,
                timestamp: new Date().toISOString(),
                predictions: data.predictions,
                is_safe: this.isSafePrediction(data.predictions, data.threshold)
            });

            // Keep only last 10 predictions
            recent = recent.slice(0, 10);
            localStorage.setItem('recentPredictions', JSON.stringify(recent));
        } catch (error) {
            console.error('Failed to save recent prediction:', error);
        }
    }

    // Animation and UI Enhancement
    initializeAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, observerOptions);

        // Observe cards and sections
        const animatedElements = document.querySelectorAll('.glass-card, .action-card, .model-card');
        animatedElements.forEach(el => observer.observe(el));
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }

    smoothScrollTo(target) {
        const element = document.querySelector(target);
        if (element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    updateActiveNavigation() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');

        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    }

    autoSaveForm(form) {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        const formId = form.id || 'autoSaveForm';

        try {
            localStorage.setItem(`autoSave_${formId}`, JSON.stringify(data));
            this.showAlert('Đã tự động lưu!', 'info', 1000);
        } catch (error) {
            console.error('Auto-save failed:', error);
        }
    }

    loadInitialData() {
        // Load system stats on supported pages
        const statsElements = document.querySelectorAll('[id$="Count"]');
        if (statsElements.length > 0) {
            this.loadSystemStats();
        }
    }

    // Cleanup
    destroy() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        // Remove event listeners
        // Note: Modern browsers handle this automatically when the page unloads
    }
}

// Initialize the application
const urlDetectionApp = new URLDetectionApp();

// Export for global access
window.URLDetectionApp = urlDetectionApp;

// Utility functions for global use
window.showAlert = (message, type, duration) => urlDetectionApp.showAlert(message, type, duration);
window.copyToClipboard = (text) => urlDetectionApp.copyToClipboard(text);
window.validateURL = (input) => urlDetectionApp.validateURL(input);

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause auto-refresh
        if (urlDetectionApp.refreshTimer) {
            clearInterval(urlDetectionApp.refreshTimer);
        }
    } else {
        // Page is visible, resume auto-refresh
        urlDetectionApp.setupAutoRefresh();
    }
});

// Handle errors gracefully
window.addEventListener('error', (event) => {
    console.error('JavaScript error:', event.error);
    if (urlDetectionApp) {
        urlDetectionApp.showAlert('Đã xảy ra lỗi không mong muốn!', 'danger');
    }
});

// Service worker registration (optional, for offline support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
