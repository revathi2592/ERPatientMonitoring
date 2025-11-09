// ER Patient Vital Monitoring - Frontend JavaScript

let conversationHistory = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadWelcomeMessage();
    setupEventListeners();
    focusInput();
});

// Load welcome message from API
async function loadWelcomeMessage() {
    try {
        const response = await fetch('/api/welcome');
        const data = await response.json();
        
        const welcomeDiv = document.getElementById('welcomeMessage');
        welcomeDiv.innerHTML = `
            <div style="max-width: 600px; margin: 0 auto;">
                <i class="fas fa-robot" style="font-size: 64px; color: var(--primary-color); margin-bottom: 20px;"></i>
                <div style="white-space: pre-line; text-align: left; line-height: 1.8;">
                    ${data.message}
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading welcome message:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    const input = document.getElementById('queryInput');
    const button = document.getElementById('sendButton');
    
    // Enter key to send
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
    
    // Button click to send
    button.addEventListener('click', sendQuery);
}

// Send query to API
async function sendQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    
    if (!query) return;
    
    // Hide welcome message
    const welcomeDiv = document.getElementById('welcomeMessage');
    if (welcomeDiv) {
        welcomeDiv.style.display = 'none';
    }
    
    // Add user message
    addMessage('user', query);
    
    // Clear input
    input.value = '';
    
    // Disable input while processing
    setInputState(false);
    
    // Show loading
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: 5
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Add agent response
        addAgentResponse(data);
        
        // Store in history
        conversationHistory.push({ query, response: data });
        
    } catch (error) {
        console.error('Error:', error);
        removeMessage(loadingId);
        addMessage('agent', `❌ Error: ${error.message}`);
    } finally {
        setInputState(true);
        focusInput();
    }
}

// Add user or agent message
function addMessage(type, content) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.id = `msg-${Date.now()}`;
    
    const avatar = type === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    const name = type === 'user' ? 'You' : 'ER Assistant';
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">${avatar}</div>
            <div class="message-name">${name}</div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
        <div class="message-content">${content}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageDiv.id;
}

// Add loading message
function addLoadingMessage() {
    const chatContainer = document.getElementById('chatContainer');
    const loadingDiv = document.createElement('div');
    const loadingId = `loading-${Date.now()}`;
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message agent';
    
    loadingDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-name">ER Assistant</div>
        </div>
        <div class="message-content loading">
            <i class="fas fa-spinner fa-spin"></i>
            <span>Analyzing patient data...</span>
        </div>
    `;
    
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return loadingId;
}

// Remove message
function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

// Add formatted agent response
function addAgentResponse(data) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    
    let contentHtml = `<div>${data.summary}</div>`;
    
    // Add vitals if available
    if (data.effects_analysis && data.effects_analysis.vitals_analyzed) {
        contentHtml += formatVitals(data.effects_analysis.vitals_analyzed);
    }
    
    // Add effects analysis if available
    if (data.effects_analysis && data.effects_analysis.conditions && data.effects_analysis.conditions.length > 0) {
        contentHtml += formatEffectsAnalysis(data.effects_analysis);
    }
    
    // Add statistics if available
    if (data.analysis && data.analysis.stats && Object.keys(data.analysis.stats).length > 0) {
        contentHtml += formatStatistics(data.analysis.stats);
    }
    
    // Add PDF link if available
    if (data.pdf_report_url) {
        contentHtml += `
            <a href="${data.pdf_report_url}" target="_blank" class="pdf-link">
                <i class="fas fa-file-pdf"></i>
                Download PDF Report
            </a>
        `;
    }
    
    // Add error if present
    if (data.error) {
        contentHtml += `<div style="color: var(--danger-color); margin-top: 10px;">⚠️ ${data.error}</div>`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-name">ER Assistant</div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
        <div class="message-content">${contentHtml}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format vitals display
function formatVitals(vitals) {
    let html = '<div class="vitals-display">';
    
    const vitalLabels = {
        'heart_rate': 'Heart Rate',
        'bp_systolic': 'BP Systolic',
        'bp_diastolic': 'BP Diastolic',
        'oxygen_level': 'Oxygen Level'
    };
    
    const vitalUnits = {
        'heart_rate': 'bpm',
        'bp_systolic': 'mmHg',
        'bp_diastolic': 'mmHg',
        'oxygen_level': '%'
    };
    
    for (const [key, value] of Object.entries(vitals)) {
        html += `
            <div class="vital-card">
                <div class="vital-label">${vitalLabels[key] || key}</div>
                <div class="vital-value">${value.toFixed(1)} <small>${vitalUnits[key] || ''}</small></div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

// Format effects analysis
function formatEffectsAnalysis(effectsData) {
    let html = `
        <div class="effects-analysis">
            <h4><i class="fas fa-exclamation-triangle"></i> Clinical Conditions Identified (${effectsData.condition_count})</h4>
    `;
    
    effectsData.conditions.forEach((condition, index) => {
        html += `
            <div class="condition-item">
                <div class="condition-name">${index + 1}. ${condition.condition}</div>
                <div class="condition-effects">
                    <strong>Potential Effects:</strong> ${condition.potential_effects}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    return html;
}

// Format statistics
function formatStatistics(stats) {
    let html = '<div style="margin-top: 15px;"><strong>📊 Statistical Summary:</strong><pre>';
    
    for (const [vital, values] of Object.entries(stats)) {
        html += `\n${vital}:\n`;
        html += `  Mean: ${values.mean.toFixed(1)}\n`;
        html += `  Min: ${values.min.toFixed(1)}\n`;
        html += `  Max: ${values.max.toFixed(1)}\n`;
        html += `  Std Dev: ${values.std.toFixed(1)}\n`;
    }
    
    html += '</pre></div>';
    return html;
}

// Insert sample query
function insertSample(text) {
    const input = document.getElementById('queryInput');
    input.value = text;
    input.focus();
}

// Set input state (enabled/disabled)
function setInputState(enabled) {
    const input = document.getElementById('queryInput');
    const button = document.getElementById('sendButton');
    
    input.disabled = !enabled;
    button.disabled = !enabled;
}

// Focus input
function focusInput() {
    document.getElementById('queryInput').focus();
}

// Update status indicator
function updateStatus(isConnected) {
    const indicator = document.getElementById('statusIndicator');
    const text = document.getElementById('statusText');
    
    if (isConnected) {
        indicator.style.background = 'var(--success-color)';
        text.textContent = 'Connected';
    } else {
        indicator.style.background = 'var(--danger-color)';
        text.textContent = 'Disconnected';
    }
}

// Check API health
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        updateStatus(data.status === 'healthy');
    } catch (error) {
        updateStatus(false);
    }
}

// Check health periodically
setInterval(checkHealth, 30000); // Every 30 seconds
checkHealth(); // Initial check
