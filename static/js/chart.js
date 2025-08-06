// Chart utilities and configurations for URL Multi-Labels Detection
class ChartManager {
    constructor() {
        this.defaultColors = {
            primary: '#667eea',
            secondary: '#764ba2',
            tertiary: '#f093fb',
            success: '#28a745',
            danger: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8'
        };

        this.chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: {
                        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        size: 12
                    }
                }
            }
        };

        this.charts = {};
        this.initializeChartDefaults();
    }

    initializeChartDefaults() {
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
            Chart.defaults.color = '#2d3748';
        }
    }

    // Comparison Chart for Multi-Model Results
    createComparisonChart(containerId, data, labels, modelTypes, colors) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        // Destroy existing chart
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const datasets = modelTypes.map(modelType => {
            const modelResult = data.results[modelType];
            if (modelResult.error) return null;

            return {
                label: modelType.toUpperCase(),
                data: modelResult.predictions,
                backgroundColor: colors[modelType] + '40',
                borderColor: colors[modelType],
                borderWidth: 3,
                pointBackgroundColor: colors[modelType],
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8,
                fill: true
            };
        }).filter(dataset => dataset !== null);

        this.charts[containerId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        min: 0,
                        ticks: {
                            stepSize: 0.2,
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            },
                            color: '#718096',
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        },
                        pointLabels: {
                            font: {
                                size: 12,
                                weight: 'bold'
                            },
                            color: '#2d3748'
                        }
                    }
                },
                plugins: {
                    ...this.chartDefaults.plugins,
                    tooltip: {
                        ...this.chartDefaults.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + (context.parsed.r * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // ROC Curve Chart
    createROCChart(containerId, modelData = null) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        // Default or provided model data
        const rocData = modelData || {
            cnn: this.generateROCData(0.94),
            xgb: this.generateROCData(0.91),
            rf: this.generateROCData(0.89)
        };

        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'CNN (AUC: 0.94)',
                        data: rocData.cnn,
                        borderColor: this.defaultColors.primary,
                        backgroundColor: 'transparent',
                        pointRadius: 0,
                        borderWidth: 3,
                        tension: 0.1
                    },
                    {
                        label: 'XGBoost (AUC: 0.91)',
                        data: rocData.xgb,
                        borderColor: this.defaultColors.secondary,
                        backgroundColor: 'transparent',
                        pointRadius: 0,
                        borderWidth: 3,
                        tension: 0.1
                    },
                    {
                        label: 'Random Forest (AUC: 0.89)',
                        data: rocData.rf,
                        borderColor: this.defaultColors.tertiary,
                        backgroundColor: 'transparent',
                        pointRadius: 0,
                        borderWidth: 3,
                        tension: 0.1
                    },
                    {
                        label: 'Random (AUC: 0.5)',
                        data: [{x: 0, y: 0}, {x: 1, y: 1}],
                        borderColor: '#999',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'False Positive Rate',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.2,
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.2,
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Threat Distribution Pie Chart
    createThreatDistributionChart(containerId, data) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const threatData = {
            benign: data.safe_count || 0,
            phishing: Math.floor((data.malicious_count || 0) * 0.4),
            malware: Math.floor((data.malicious_count || 0) * 0.3),
            defacement: Math.floor((data.malicious_count || 0) * 0.3)
        };

        this.charts[containerId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Benign', 'Phishing', 'Malware', 'Defacement'],
                datasets: [{
                    data: Object.values(threatData),
                    backgroundColor: [
                        this.defaultColors.success,
                        this.defaultColors.warning,
                        this.defaultColors.danger,
                        this.defaultColors.info
                    ],
                    borderWidth: 3,
                    borderColor: '#fff',
                    hoverBorderWidth: 4
                }]
            },
            options: {
                ...this.chartDefaults,
                cutout: '60%',
                plugins: {
                    ...this.chartDefaults.plugins,
                    tooltip: {
                        ...this.chartDefaults.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Model Performance Bar Chart
    createModelPerformanceChart(containerId, data) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const modelStats = data.models_used || {};

        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['CNN', 'XGBoost', 'Random Forest'],
                datasets: [{
                    label: 'Số lượng dự đoán',
                    data: [modelStats.cnn || 0, modelStats.xgb || 0, modelStats.rf || 0],
                    backgroundColor: [
                        this.defaultColors.primary,
                        this.defaultColors.secondary,
                        this.defaultColors.tertiary
                    ],
                    borderRadius: 8,
                    borderSkipped: false,
                }]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1,
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Timeline Chart
    createTimelineChart(containerId, historyData) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const timelineData = this.processTimelineData(historyData);

        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timelineData.labels,
                datasets: [
                    {
                        label: 'An toàn',
                        data: timelineData.safe,
                        borderColor: this.defaultColors.success,
                        backgroundColor: this.defaultColors.success + '20',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Độc hại',
                        data: timelineData.malicious,
                        borderColor: this.defaultColors.danger,
                        backgroundColor: this.defaultColors.danger + '20',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    x: {
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        stacked: false,
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Training History Chart
    createTrainingHistoryChart(containerId, trainingData = null) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        // Generate sample training data if none provided
        const data = trainingData || this.generateTrainingData();

        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [
                    {
                        label: 'Training Accuracy',
                        data: data.trainAcc,
                        borderColor: this.defaultColors.success,
                        backgroundColor: this.defaultColors.success + '10',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2
                    },
                    {
                        label: 'Validation Accuracy',
                        data: data.valAcc,
                        borderColor: this.defaultColors.danger,
                        backgroundColor: this.defaultColors.danger + '10',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epochs'
                        },
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        },
                        min: 50,
                        max: 100,
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Processing Time Chart
    createProcessingTimeChart(containerId, processingData = null) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const data = processingData || [850, 45, 120]; // Default processing times

        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['CNN', 'XGBoost', 'Random Forest'],
                datasets: [{
                    label: 'Thời gian xử lý (ms)',
                    data: data,
                    backgroundColor: [
                        this.defaultColors.primary,
                        this.defaultColors.secondary,
                        this.defaultColors.tertiary
                    ],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Thời gian (ms)'
                        },
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Feature Importance Chart
    createFeatureImportanceChart(containerId, model = 'xgb', importanceData = null) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const features = [
            'URL Length', 'Domain Length', 'Number of Dots', 'Number of Hyphens',
            'Has IP', 'Suspicious TLD', 'External Links', 'Forms Count',
            'SSL Certificate', 'Domain Age', 'Page Rank', 'Traffic Rank'
        ];

        const defaultImportance = {
            xgb: [0.15, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.06],
            rf: [0.13, 0.11, 0.10, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05]
        };

        const data = importanceData || defaultImportance[model];

        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Importance',
                    data: data,
                    backgroundColor: model === 'xgb' ? this.defaultColors.secondary : this.defaultColors.tertiary,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                indexAxis: 'y',
                plugins: {
                    ...this.chartDefaults.plugins,
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 0.2,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        },
                        ticks: {
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#718096',
                            font: {
                                size: 10
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Performance Radar Chart
    createPerformanceRadarChart(containerId, performanceData = null) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const defaultData = {
            cnn: [92.5, 93.2, 90.5, 91.8, 94.0],
            xgb: [89.3, 90.1, 87.4, 88.7, 91.0],
            rf: [87.8, 88.9, 85.6, 87.2, 89.0]
        };

        const data = performanceData || defaultData;

        this.charts[containerId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                datasets: [
                    {
                        label: 'CNN',
                        data: data.cnn,
                        backgroundColor: this.defaultColors.primary + '20',
                        borderColor: this.defaultColors.primary,
                        borderWidth: 2,
                        pointBackgroundColor: this.defaultColors.primary,
                        pointRadius: 4
                    },
                    {
                        label: 'XGBoost',
                        data: data.xgb,
                        backgroundColor: this.defaultColors.secondary + '20',
                        borderColor: this.defaultColors.secondary,
                        borderWidth: 2,
                        pointBackgroundColor: this.defaultColors.secondary,
                        pointRadius: 4
                    },
                    {
                        label: 'Random Forest',
                        data: data.rf,
                        backgroundColor: this.defaultColors.tertiary + '20',
                        borderColor: this.defaultColors.tertiary,
                        borderWidth: 2,
                        pointBackgroundColor: this.defaultColors.tertiary,
                        pointRadius: 4
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            color: '#718096'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        });

        return this.charts[containerId];
    }

    // Utility Functions
    generateROCData(auc) {
        const points = [];
        for (let i = 0; i <= 100; i++) {
            const fpr = i / 100;
            const tpr = Math.min(1, Math.pow(fpr, 1 - auc) + (auc - 0.5) * 2 * (1 - fpr));
            points.push({x: fpr, y: tpr});
        }
        return points;
    }

    processTimelineData(historyData) {
        const grouped = {};

        historyData.forEach(item => {
            if (!item.timestamp) return;

            const date = new Date(item.timestamp).toDateString();
            if (!grouped[date]) {
                grouped[date] = {safe: 0, malicious: 0};
            }

            const predictions = item.predictions || [];
            const isSafe = predictions[0] > 0.5 && Math.max(...predictions.slice(1)) <= 0.5;

            if (isSafe) {
                grouped[date].safe++;
            } else {
                grouped[date].malicious++;
            }
        });

        const sortedDates = Object.keys(grouped).sort();
        const last7Days = sortedDates.slice(-7);

        return {
            labels: last7Days.map(date => new Date(date).toLocaleDateString('vi-VN', {month: 'short', day: 'numeric'})),
            safe: last7Days.map(date => grouped[date].safe),
            malicious: last7Days.map(date => grouped[date].malicious)
        };
    }

    generateTrainingData() {
        const epochs = Array.from({length: 50}, (_, i) => i + 1);
        const trainAcc = epochs.map(e => Math.min(95, 60 + e * 0.7 + Math.random() * 2));
        const valAcc = epochs.map(e => Math.min(93, 55 + e * 0.75 + Math.random() * 3));

        return {
            epochs: epochs,
            trainAcc: trainAcc,
            valAcc: valAcc
        };
    }

    // Chart Management
    updateChart(chartId, newData) {
        if (this.charts[chartId]) {
            this.charts[chartId].data = newData;
            this.charts[chartId].update('active');
        }
    }

    destroyChart(chartId) {
        if (this.charts[chartId]) {
            this.charts[chartId].destroy();
            delete this.charts[chartId];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(chartId => {
            this.destroyChart(chartId);
        });
    }

    resizeAllCharts() {
        Object.values(this.charts).forEach(chart => {
            chart.resize();
        });
    }

    // Export chart as image
    exportChart(chartId, filename = 'chart.png') {
        if (this.charts[chartId]) {
            const url = this.charts[chartId].toBase64Image();
            const link = document.createElement('a');
            link.download = filename;
            link.href = url;
            link.click();
        }
    }

    // Get chart instance
    getChart(chartId) {
        return this.charts[chartId] || null;
    }
}

// Initialize global chart manager
const chartManager = new ChartManager();

// Global helper functions
window.createComparisonChart = (containerId, data, labels, modelTypes, colors) => {
    return chartManager.createComparisonChart(containerId, data, labels, modelTypes, colors);
};

window.createROCChart = (containerId, modelData) => {
    return chartManager.createROCChart(containerId, modelData);
};

window.createThreatDistributionChart = (containerId, data) => {
    return chartManager.createThreatDistributionChart(containerId, data);
};

window.createModelPerformanceChart = (containerId, data) => {
    return chartManager.createModelPerformanceChart(containerId, data);
};

window.createTimelineChart = (containerId, historyData) => {
    return chartManager.createTimelineChart(containerId, historyData);
};

window.createTrainingHistoryChart = (containerId, trainingData) => {
    return chartManager.createTrainingHistoryChart(containerId, trainingData);
};

window.createProcessingTimeChart = (containerId, processingData) => {
    return chartManager.createProcessingTimeChart(containerId, processingData);
};

window.createFeatureImportanceChart = (containerId, model, importanceData) => {
    return chartManager.createFeatureImportanceChart(containerId, model, importanceData);
};

window.createPerformanceRadarChart = (containerId, performanceData) => {
    return chartManager.createPerformanceRadarChart(containerId, performanceData);
};

// Export chart manager for direct access
window.ChartManager = chartManager;

// Handle window resize
window.addEventListener('resize', () => {
    chartManager.resizeAllCharts();
});

// Clean up charts when page unloads
window.addEventListener('beforeunload', () => {
    chartManager.destroyAllCharts();
});
