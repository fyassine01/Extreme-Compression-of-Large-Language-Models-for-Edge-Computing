// Global state
let totalQueries = 0;
let agreementCount = 0;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    const textInput = document.getElementById('textInput');
    const sendButton = document.getElementById('sendButton');

    // Send on Enter (but allow Shift+Enter for new line)
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Enable/disable button based on input
    textInput.addEventListener('input', function() {
        sendButton.disabled = !textInput.value.trim();
    });

    // Initial state
    sendButton.disabled = !textInput.value.trim();
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model_info');
        const data = await response.json();

        if (data.teacher && data.teacher.parameters) {
            document.getElementById('teacher-params').textContent = data.teacher.parameters;
        }

        if (data.student && data.student.parameters) {
            document.getElementById('student-params').textContent = data.student.parameters;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Send message and get predictions
async function sendMessage() {
    const textInput = document.getElementById('textInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const text = textInput.value.trim();

    if (!text) {
        return;
    }

    // Disable button and show loading
    sendButton.disabled = true;
    sendButton.innerHTML = '<span class="loading"></span> Analyzing...';

    // Remove welcome message if present
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // Add user message
    addUserMessage(text);
    textInput.value = '';
    textInput.focus();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (response.ok && !data.error) {
            addComparisonMessage(data);
            updateStats(data);
            totalQueries++;
            if (data.comparison && data.comparison.agreement) {
                agreementCount++;
            }
        } else {
            addErrorMessage(data.error || 'An error occurred while analyzing the text.');
        }
    } catch (error) {
        console.error('Error:', error);
        addErrorMessage('Network error: Could not connect to the server.');
    } finally {
        // Re-enable button
        sendButton.disabled = false;
        sendButton.innerHTML = '<span>Analyze</span><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
        sendButton.disabled = !textInput.value.trim();
    }
}

// Add user message to chat
function addUserMessage(text) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';

    messageDiv.innerHTML = `
        <div class="message-header">
            <span>👤 You</span>
        </div>
        <div class="message-text">${escapeHtml(text)}</div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add comparison message with predictions
function addComparisonMessage(data) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';

    const teacher = data.teacher;
    const student = data.student;
    const comparison = data.comparison;

    let comparisonHtml = `
        <div class="message-header">
            <span>🤖 Model Predictions</span>
        </div>
        <div class="comparison-container">
            ${createModelCard('teacher', teacher, 'BERT-base')}
            ${createModelCard('student', student, 'DistilBERT (Compressed)')}
        </div>
    `;

    // Add comparison note if predictions differ
    if (comparison && !comparison.agreement) {
        comparisonHtml += `
            <div class="error-message" style="margin-top: 15px;">
                ⚠️ Models disagree: BERT predicts "${teacher.sentiment}" while DistilBERT predicts "${student.sentiment}"
            </div>
        `;
    }

    messageDiv.innerHTML = comparisonHtml;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Create model prediction card
function createModelCard(modelType, result, modelName) {
    if (result.error) {
        return `
            <div class="comparison-card ${modelType}">
                <div class="card-header">
                    <div class="card-title">${modelName}</div>
                </div>
                <div class="error-message">
                    Error: ${result.error}
                </div>
            </div>
        `;
    }

    const sentiment = result.sentiment;
    const confidence = result.confidence;
    const latency = result.latency_ms;
    const probabilities = result.probabilities || {};

    const sentimentClass = sentiment.toLowerCase();
    const positiveProb = probabilities.positive || 0;
    const negativeProb = probabilities.negative || 0;

    return `
        <div class="comparison-card ${modelType}">
            <div class="card-header">
                <div class="card-title">${modelName}</div>
                <div class="latency-badge">${latency} ms</div>
            </div>
            <div class="prediction-result">
                <div class="sentiment-label ${sentimentClass}">${sentiment}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span>Confidence</span>
                        <span>${confidence}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${sentimentClass}" style="width: ${confidence}%"></div>
                    </div>
                </div>
                <div class="probabilities">
                    <div class="prob-item">
                        <div class="prob-label">Negative</div>
                        <div class="prob-value">${negativeProb}%</div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-label">Positive</div>
                        <div class="prob-value">${positiveProb}%</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Add error message
function addErrorMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';

    messageDiv.innerHTML = `
        <div class="error-message">
            ❌ ${escapeHtml(message)}
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Update comparison statistics
function updateStats(data) {
    const statsDiv = document.getElementById('comparisonStats');
    const speedupEl = document.getElementById('speedup');
    const agreementEl = document.getElementById('agreement');
    const totalQueriesEl = document.getElementById('totalQueries');

    if (statsDiv.style.display === 'none') {
        statsDiv.style.display = 'block';
    }

    if (data.comparison && data.comparison.speedup) {
        speedupEl.textContent = data.comparison.speedup + 'x';
        speedupEl.style.color = '#10b981';
    } else {
        speedupEl.textContent = '-';
    }

    if (data.comparison) {
        const agreementPercent = totalQueries > 0 
            ? Math.round((agreementCount / totalQueries) * 100) 
            : (data.comparison.agreement ? 100 : 0);
        agreementEl.textContent = agreementPercent + '%';
        agreementEl.style.color = agreementPercent >= 90 ? '#10b981' : 
                                  agreementPercent >= 70 ? '#f59e0b' : '#ef4444';
    }

    totalQueriesEl.textContent = totalQueries;
}

// Scroll to bottom of chat
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

