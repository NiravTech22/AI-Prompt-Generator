// Content script for AI Prompt Generator Extension
// This script runs on all web pages and can interact with page content

console.log('AI Prompt Generator extension loaded');

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'insertPrompt') {
        insertPromptIntoPage(request.prompt);
        sendResponse({success: true});
    }
});

function insertPromptIntoPage(prompt) {
    // Find common text input elements on the page
    const selectors = [
        'textarea',
        'input[type="text"]',
        '[contenteditable="true"]',
        '.ql-editor', // Quill editor
        '.DraftEditor-root', // Draft.js editor
        '.CodeMirror textarea', // CodeMirror
        '[data-testid="textbox"]', // Common test ID
        '.input-field',
        '.text-input',
        '#prompt-input',
        '#chat-input'
    ];

    let targetElement = null;
    
    // Find the first focusable text input
    for (const selector of selectors) {
        const elements = document.querySelectorAll(selector);
        for (const element of elements) {
            if (isVisible(element) && !element.disabled && !element.readOnly) {
                targetElement = element;
                break;
            }
        }
        if (targetElement) break;
    }

    if (targetElement) {
        // Focus the element
        targetElement.focus();
        
        // Insert the prompt
        if (targetElement.tagName === 'TEXTAREA' || targetElement.tagName === 'INPUT') {
            targetElement.value = prompt;
            targetElement.dispatchEvent(new Event('input', { bubbles: true }));
        } else if (targetElement.contentEditable === 'true') {
            targetElement.textContent = prompt;
            targetElement.dispatchEvent(new Event('input', { bubbles: true }));
        }
        
        // Trigger change events
        targetElement.dispatchEvent(new Event('change', { bubbles: true }));
        
        // Show success indicator
        showInsertionSuccess(targetElement);
    } else {
        // Show notification if no suitable input found
        showNotification('No text input found on this page. Copy the prompt manually from the extension popup.');
    }
}

function isVisible(element) {
    const style = window.getComputedStyle(element);
    return style.display !== 'none' && 
           style.visibility !== 'hidden' && 
           style.opacity !== '0' &&
           element.offsetWidth > 0 && 
           element.offsetHeight > 0;
}

function showInsertionSuccess(element) {
    const indicator = document.createElement('div');
    indicator.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #4CAF50;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease-out;
    `;
    
    indicator.innerHTML = '✅ Prompt inserted successfully!';
    document.body.appendChild(indicator);
    
    // Add animation keyframes
    if (!document.getElementById('prompt-generator-styles')) {
        const style = document.createElement('style');
        style.id = 'prompt-generator-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    setTimeout(() => {
        indicator.remove();
    }, 3000);
}

function showNotification(message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ff9800;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Optional: Add keyboard shortcut for quick access
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Shift + P to open extension
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'P') {
        e.preventDefault();
        chrome.runtime.sendMessage({action: 'openPopup'});
    }
});