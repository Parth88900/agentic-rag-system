document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;
    
    // Add user message
    addMessage(text, 'user');
    input.value = '';
    
    // Show typing indicator
    const typingId = showTyping();
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query: text })
        });
        
        const data = await response.json();
        removeTyping(typingId);
        
        if (response.ok) {
            addSystemMessage(data);
        } else {
            addMessage("Error: " + (data.detail || "Something went wrong"), 'system');
        }
    } catch (err) {
        removeTyping(typingId);
        addMessage("Connection error while reaching the AI.", 'system');
    }
});

function addMessage(text, sender) {
    const messages = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    msgDiv.innerHTML = `<div class="bubble"><p>${escapeHtml(text)}</p></div>`;
    messages.appendChild(msgDiv);
    messages.scrollTop = messages.scrollHeight;
}

function addSystemMessage(data) {
    const messages = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message system`;
    
    // Format text preserving newlines
    let formattedText = escapeHtml(data.answer).replace(/\n/g, '<br>');
    
    // Format confidence label
    const confScore = Math.round(data.confidence * 100);

    let metaHtml = `
    <div class="meta-info">
        <span class="tag">Type: ${data.query_type}</span>
        <span class="tag">Conf: ${confScore}%</span>
        <span class="tag">Via: ${data.generation_method}</span>
    </div>`;
    
    // Check if the answer already includes a confidence indicator at the top
    formattedText = formattedText.replace(/\[HIGH CONFIDENCE\]/gi, '<strong>[High Confidence]</strong><br>');
    formattedText = formattedText.replace(/\[MODERATE CONFIDENCE\]/gi, '<strong>[Moderate Confidence]</strong><br>');

    msgDiv.innerHTML = `<div class="bubble"><p>${formattedText}</p></div>${metaHtml}`;
    messages.appendChild(msgDiv);
    messages.scrollTop = messages.scrollHeight;
}

function showTyping() {
    const messages = document.getElementById('chat-messages');
    const id = 'typing-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message system typing`;
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="bubble">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    messages.appendChild(msgDiv);
    messages.scrollTop = messages.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
