// Simple Multi-Model Prediction JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Multi-model prediction script loaded');
    
    // Handle form submission
    const form = document.getElementById('multiModelForm');
    if (form) {
        form.addEventListener('submit', handleMultiModelPrediction);
    }
    
    // Handle threshold slider
    const thresholdRange = document.getElementById('thresholdRange');
    const thresholdValue = document.getElementById('thresholdValue');
    if (thresholdRange && thresholdValue) {
        thresholdRange.addEventListener('input', function() {
            thresholdValue.textContent = this.value + '%';
        });
    }
});

async function handleMultiModelPrediction(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const url = formData.get('url');
    const threshold = parseFloat(formData.get('threshold')) / 100 || 0.5;
    
    console.log('Form submission:', { url, threshold });
    
    // Validate URL
    if (!url || !url.trim()) {
        showAlert('Vui lòng nhập URL cần kiểm tra!', 'danger');
        return;
    }
    
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        showAlert('URL phải bắt đầu với http:// hoặc https://!', 'danger');
        return;
    }
    
    // Show loading
    showLoading();
    hideResults();
    
    try {
        const requestData = {
            url: url.trim(),
            threshold: threshold
        };
        
        console.log('Sending request:', requestData);
        
        const response = await fetch('/api/predict-multi-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            displayResults(data);
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        showAlert('Có lỗi xảy ra khi phân tích URL! Vui lòng thử lại.', 'danger');
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    console.log('Displaying results:', data);
    
    if (!data.comparison_results || !Array.isArray(data.comparison_results)) {
        console.error('Invalid response format:', data);
        showAlert('Dữ liệu phản hồi không hợp lệ!', 'danger');
        return;
    }
    
    // Model mapping
    const modelMapping = {
        'CNN_NUM': { key: 'cnn', name: 'CNN (Numerical)' },
        'CNN_NON': { key: 'cnn', name: 'CNN (Non-numerical)' },
        'XGB_NUM': { key: 'xgb', name: 'XGBoost (Numerical)' },
        'XGB_NON': { key: 'xgb', name: 'XGBoost (Non-numerical)' },
        'RF_NUM': { key: 'rf', name: 'Random Forest (Numerical)' },
        'RF_NON': { key: 'rf', name: 'Random Forest (Non-numerical)' },
        'BERT_NON': { key: 'bert', name: 'BERT (Non-numerical)' }
    };
    
    // Process results
    const processedResults = {};
    
    data.comparison_results.forEach(result => {
        const mapping = modelMapping[result.model_name];
        if (!mapping) return;
        
        const modelKey = mapping.key;
        
        // Prefer non-numerical results or use first available
        if (!processedResults[modelKey] || result.model_name.endsWith('_NON')) {
            processedResults[modelKey] = {
                name: mapping.name,
                probabilities: result.probabilities,
                predicted_labels: result.predicted_labels,
                error: result.error
            };
        }
    });
    
    console.log('Processed results:', processedResults);
    
    // Update model cards
    const modelTypes = ['cnn', 'xgb', 'rf', 'bert'];
    const labelNames = ['benign', 'defacement', 'malware', 'phishing'];
    
    modelTypes.forEach(modelType => {
        const resultCard = document.getElementById(`${modelType}Results`);
        if (!resultCard) return;
        
        const modelContent = resultCard.querySelector('.model-content');
        if (!modelContent) return;
        
        const result = processedResults[modelType];
        
        if (!result) {
            modelContent.innerHTML = '<p class="text-muted">Không có kết quả</p>';
            return;
        }
        
        if (result.error) {
            modelContent.innerHTML = `<div class="alert alert-danger">Lỗi: ${result.error}</div>`;
            return;
        }
        
        if (!result.probabilities || !Array.isArray(result.probabilities)) {
            modelContent.innerHTML = '<p class="text-warning">Dữ liệu không hợp lệ</p>';
            return;
        }
        
        // Generate prediction bars
        let barsHTML = '';
        result.probabilities.forEach((prob, index) => {
            const percentage = (prob * 100).toFixed(1);
            const isHigh = prob > (data.threshold || 0.5);
            
            barsHTML += `
                <div class="prediction-item mb-2">
                    <div class="d-flex justify-content-between">
                        <small>${labelNames[index]}</small>
                        <small class="fw-bold ${isHigh ? 'text-danger' : 'text-secondary'}">${percentage}%</small>
                    </div>
                    <div class="progress" style="height: 4px;">
                        <div class="progress-bar ${isHigh ? 'bg-danger' : 'bg-secondary'}"
                             style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        });
        
        // Determine safety
        const benignProb = result.probabilities[0] || 0;
        const maliciousProbs = result.probabilities.slice(1);
        const maxMalicious = Math.max(...maliciousProbs);
        const isSafe = benignProb > (data.threshold || 0.5) && maxMalicious <= (data.threshold || 0.5);
        
        // Update card appearance
        resultCard.className = `model-result-card ${isSafe ? 'safe' : 'danger'}`;
        
        modelContent.innerHTML = `
            <div class="model-header">
                <h6>${result.name}</h6>
                <span class="badge ${isSafe ? 'bg-success' : 'bg-danger'}">
                    ${isSafe ? 'An toàn' : 'Nguy hiểm'}
                </span>
            </div>
            <div class="predictions-container">
                ${barsHTML}
            </div>
        `;
    });
    
    // Show results
    showResults();
    
    console.log('Results displayed successfully');
}

function showLoading() {
    const loadingSection = document.getElementById('loadingSection');
    if (loadingSection) {
        loadingSection.style.display = 'flex';
    }
}

function hideLoading() {
    const loadingSection = document.getElementById('loadingSection');
    if (loadingSection) {
        loadingSection.style.display = 'none';
    }
}

function showResults() {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
}

function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
}

function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <strong>${type === 'danger' ? 'Lỗi!' : 'Thông báo!'}</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert alert
    const container = document.querySelector('.form-container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
    }
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

console.log('Multi-model prediction script initialized');
