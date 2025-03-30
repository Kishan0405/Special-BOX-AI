document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------
    // CUSTOM ALERT/CONFIRM CSS & FUNCTIONS
    // -------------------------------
    const customAlertStyles = `
    .custom-alert-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.4);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .custom-alert-modal {
        background: #fff;
        padding: 20px;
        border-radius: 5px;
        max-width: 90%;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .custom-alert-buttons {
        margin-top: 15px;
    }
    .custom-alert-buttons button {
        margin: 0 10px;
        padding: 8px 16px;
        border: none;
        border-radius: 3px;
        cursor: pointer;
    }
    .custom-alert-ok-btn {
        background-color: #007bff;
        color: #fff;
    }
    .custom-alert-cancel-btn {
        background-color: #6c757d;
        color: #fff;
    }
    /* -------------------------------
       TYPING INDICATOR STYLES
       ------------------------------- */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 10px;
    }
    .typing-indicator .dot {
        width: 8px;
        height: 8px;
        background: #999;
        border-radius: 50%;
        animation: blink 1.4s infinite both;
    }
    .typing-indicator .dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-indicator .dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes blink {
        0% { opacity: 0.2; }
        20% { opacity: 1; }
        100% { opacity: 0.2; }
    }`;

    // Add styles once to document
    const styleSheet = document.createElement('style');
    styleSheet.type = 'text/css';
    styleSheet.innerText = customAlertStyles;
    document.head.appendChild(styleSheet);

    // Custom alert that shows a message with an OK button
    function customAlert(message) {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'custom-alert-overlay';

            const modal = document.createElement('div');
            modal.className = 'custom-alert-modal';

            const messageDiv = document.createElement('div');
            messageDiv.textContent = message;

            const buttonDiv = document.createElement('div');
            buttonDiv.className = 'custom-alert-buttons';

            const okBtn = document.createElement('button');
            okBtn.textContent = 'OK';
            okBtn.className = 'custom-alert-ok-btn';
            okBtn.addEventListener('click', () => {
                document.body.removeChild(overlay);
                resolve();
            });

            buttonDiv.appendChild(okBtn);
            modal.appendChild(messageDiv);
            modal.appendChild(buttonDiv);
            overlay.appendChild(modal);
            document.body.appendChild(overlay);
        });
    }

    // Custom confirm that shows a message with Cancel/OK buttons and returns a boolean
    function customConfirm(message) {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'custom-alert-overlay';

            const modal = document.createElement('div');
            modal.className = 'custom-alert-modal';

            const messageDiv = document.createElement('div');
            messageDiv.textContent = message;

            const buttonDiv = document.createElement('div');
            buttonDiv.className = 'custom-alert-buttons';

            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = 'Cancel';
            cancelBtn.className = 'custom-alert-cancel-btn';
            cancelBtn.addEventListener('click', () => {
                document.body.removeChild(overlay);
                resolve(false);
            });

            const okBtn = document.createElement('button');
            okBtn.textContent = 'OK';
            okBtn.className = 'custom-alert-ok-btn';
            okBtn.addEventListener('click', () => {
                document.body.removeChild(overlay);
                resolve(true);
            });

            buttonDiv.appendChild(cancelBtn);
            buttonDiv.appendChild(okBtn);
            modal.appendChild(messageDiv);
            modal.appendChild(buttonDiv);
            overlay.appendChild(modal);
            document.body.appendChild(overlay);
        });
    }

    // -------------------------------
    // DOM ELEMENTS - Cache all selectors
    // -------------------------------
    const elements = {
        agentSelect: document.getElementById('agent'),
        responseTypeSelect: document.getElementById('response-type'),
        toneSelect: document.getElementById('tone'),
        languageSelect: document.getElementById('language'),
        chatMessages: document.getElementById('chat-messages'),
        userInput: document.getElementById('user-input'),
        sendBtn: document.getElementById('send-btn'),
        resetBtn: document.getElementById('reset-btn'),
        spinner: document.getElementById('spinner'),
        statusElement: document.getElementById('status'),
        chatHistoryList: document.getElementById('chat-history-list'),
        newChatBtn: document.getElementById('new-chat-btn')
    };

    // -------------------------------
    // AGENT-SPECIFIC OPTIONS
    // -------------------------------
    const options = {
        default_agent: {
            responseTypes: ['short', 'medium', 'detailed', 'detailed with references', 'bullet points'],
            tones: ['professional', 'academic', 'friendly', 'creative']
        },
        summarize_agent: {
            responseTypes: ['short', 'medium', 'detailed', 'detailed with references', 'bullet points'],
            tones: ['professional', 'academic', 'friendly', 'creative']
        },
        email_agent: {
            responseTypes: ['short', 'medium', 'detailed'],
            tones: ['professional', 'academic', 'friendly', 'creative']
        },
        coding_agent: {
            responseTypes: ['medium', 'detailed'],
            tones: ['professional']
        },
        mathematics_agent: {
            responseTypes: ['short', 'medium', 'detailed', 'bullet points'],
            tones: ['professional']
        }
    };

    // -------------------------------
    // STORAGE FUNCTIONS - Using localStorage
    // -------------------------------
    const Storage = {
        getConversations() {
            return JSON.parse(localStorage.getItem('savedConversations')) || [];
        },

        saveConversations(conversations) {
            localStorage.setItem('savedConversations', JSON.stringify(conversations));
        },

        getActiveId() {
            return localStorage.getItem('activeConversationId');
        },

        setActiveId(id) {
            localStorage.setItem('activeConversationId', id);
        },

        getConversationById(id) {
            return this.getConversations().find(conv => conv.id === id);
        },

        updateConversation(updatedConversation) {
            const conversations = this.getConversations();
            const index = conversations.findIndex(c => c.id === updatedConversation.id);

            if (index !== -1) {
                conversations[index] = updatedConversation;
            } else {
                conversations.push(updatedConversation);
            }

            this.saveConversations(conversations);
        },

        deleteConversation(id) {
            const conversations = this.getConversations().filter(c => c.id !== id);
            this.saveConversations(conversations);

            if (this.getActiveId() === id) {
                this.setActiveId(null);
            }
        }
    };

    // -------------------------------
    // CONVERSATION FUNCTIONS
    // -------------------------------
    const Conversation = {
        create() {
            return {
                id: `conv_${Date.now()}`,
                title: 'New Chat',
                messages: [],
                lastUpdate: Date.now()
            };
        },

        generateTitle(messages) {
            const firstUserMessage = messages.find(msg => msg.isUser);
            if (firstUserMessage) {
                const messageText = firstUserMessage.message;
                return messageText.length > 50 ? `${messageText.substring(0, 50)}...` : messageText;
            }
            return 'New Chat';
        },

        updateTitle(conversation) {
            if (!conversation) return;

            conversation.title = this.generateTitle(conversation.messages);
            Storage.updateConversation(conversation);
        }
    };

    // -------------------------------
    // UI FUNCTIONS
    // -------------------------------
    const UI = {
        capitalizeFirst(str) {
            return str.charAt(0).toUpperCase() + str.slice(1);
        },

        initializeDropdowns() {
            const selectedAgent = elements.agentSelect.value;
            const selectedOptions = options[selectedAgent];
            if (!selectedOptions) return;

            // Clear existing options
            elements.responseTypeSelect.innerHTML = '';
            elements.toneSelect.innerHTML = '';

            // Add response type options
            selectedOptions.responseTypes.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = UI.capitalizeFirst(type);
                elements.responseTypeSelect.appendChild(option);
            });

            // Add tone options
            selectedOptions.tones.forEach(tone => {
                const option = document.createElement('option');
                option.value = tone;
                option.textContent = UI.capitalizeFirst(tone);
                elements.toneSelect.appendChild(option);
            });
        },

        renderConversationList() {
            const conversations = Storage.getConversations();
            const activeId = Storage.getActiveId();

            elements.chatHistoryList.innerHTML = '';

            // Sort conversations by last update (newest first)
            conversations.sort((a, b) => b.lastUpdate - a.lastUpdate);

            // Create document fragment for better performance
            const fragment = document.createDocumentFragment();

            conversations.forEach((conv) => {
                const li = document.createElement('li');

                // Title element with edit functionality
                const titleSpan = document.createElement('span');
                titleSpan.textContent = conv.title;
                titleSpan.className = 'conversation-title';

                // Make title editable on double-click
                titleSpan.addEventListener('dblclick', () => {
                    titleSpan.contentEditable = 'true';
                    titleSpan.focus();
                });

                titleSpan.addEventListener('blur', () => {
                    titleSpan.contentEditable = 'false';
                    const newTitle = titleSpan.textContent.trim();
                    if (newTitle && newTitle !== conv.title) {
                        conv.title = newTitle;
                        Storage.updateConversation(conv);
                    } else {
                        titleSpan.textContent = conv.title;
                    }
                });

                titleSpan.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        titleSpan.blur();
                    }
                });

                li.appendChild(titleSpan);

                // Delete button
                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'X';
                deleteButton.className = 'delete-conversation-btn';
                deleteButton.addEventListener('click', async (event) => {
                    event.stopPropagation();
                    if (await customConfirm('Are you sure you want to delete this conversation?')) {
                        Storage.deleteConversation(conv.id);
                        UI.renderConversationList();
                        UI.renderActiveConversation();
                    }
                });
                li.appendChild(deleteButton);

                // Highlight active conversation
                if (conv.id === activeId) {
                    li.classList.add('active-conversation');
                }

                // Click to select conversation
                li.addEventListener('click', () => {
                    Storage.setActiveId(conv.id);
                    UI.renderActiveConversation();
                    UI.renderConversationList();
                });

                fragment.appendChild(li);
            });

            elements.chatHistoryList.appendChild(fragment);
        },

        renderActiveConversation() {
            elements.chatMessages.innerHTML = '';

            const activeId = Storage.getActiveId();

            // If no active conversation, create a new one
            if (!activeId) {
                const newConv = Conversation.create();
                Storage.updateConversation(newConv);
                Storage.setActiveId(newConv.id);
                return UI.renderActiveConversation();
            }

            const conversation = Storage.getConversationById(activeId);

            // If conversation not found, create a new one
            if (!conversation) {
                const newConv = Conversation.create();
                Storage.updateConversation(newConv);
                Storage.setActiveId(newConv.id);
                return UI.renderActiveConversation();
            }

            // Show greeting for empty conversations
            if (conversation.messages.length === 0) {
                const greeting = "Hello! I'm Special Box AI. How can I assist you today?";
                Chat.addMessage(greeting, false, false);
            } else {
                // Create document fragment for better performance
                const fragment = document.createDocumentFragment();

                conversation.messages.forEach(msg => {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${msg.isUser ? 'user-message' : 'ai-message'}`;

                    if (msg.isHtml && !msg.isUser) {
                        messageDiv.innerHTML = msg.message;
                    } else {
                        messageDiv.textContent = msg.message;
                    }

                    fragment.appendChild(messageDiv);
                });

                elements.chatMessages.appendChild(fragment);
                elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
            }
        },

        displayMessage(message, isUser = false, isHtml = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;

            if (isHtml && !isUser) {
                messageDiv.innerHTML = message;
            } else {
                messageDiv.textContent = message;
            }

            elements.chatMessages.appendChild(messageDiv);
            elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        },

        setLoading(isLoading) {
            elements.userInput.disabled = isLoading;
            elements.sendBtn.disabled = isLoading;
            elements.spinner.style.display = isLoading ? 'inline-block' : 'none';
            if (!isLoading) {
                elements.statusElement.textContent = '';
            }
        },

        updateStatus(status) {
            elements.statusElement.textContent = status;
        },

        // New: Show typing indicator (animated dots)
        showTypingIndicator() {
            let typingIndicator = document.getElementById('typing-indicator');
            if (!typingIndicator) {
                typingIndicator = document.createElement('div');
                typingIndicator.id = 'typing-indicator';
                typingIndicator.className = 'message ai-message typing-indicator';
                typingIndicator.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
                elements.chatMessages.appendChild(typingIndicator);
                elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
            }
        },

        // New: Hide typing indicator
        hideTypingIndicator() {
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
    };

    // -------------------------------
    // CHAT FUNCTIONS
    // -------------------------------
    const Chat = {
        addMessage(message, isUser, isHtml) {
            const activeId = Storage.getActiveId();
            if (!activeId) return;

            const conversation = Storage.getConversationById(activeId);
            if (!conversation) return;

            conversation.messages.push({ message, isUser, isHtml });
            conversation.lastUpdate = Date.now();

            Conversation.updateTitle(conversation);
            Storage.updateConversation(conversation);

            UI.displayMessage(message, isUser, isHtml);
            UI.renderConversationList();
        },

        sendMessage() {
            const message = elements.userInput.value.trim();
            if (!message) {
                customAlert('Please enter a valid message.');
                return;
            }

            Chat.addMessage(message, true, false);
            elements.userInput.value = '';
            UI.setLoading(true);
            UI.updateStatus('Starting...');
            UI.showTypingIndicator(); // Start typing animation

            const payload = {
                message,
                response_type: elements.responseTypeSelect.value,
                tone: elements.toneSelect.value,
                agent: elements.agentSelect.value,
                language: elements.languageSelect.value
            };

            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP Error! Status: ${response.status}`);
                    if (!response.body) throw new Error('ReadableStream not supported.');
                    return response.body.getReader();
                })
                .then(reader => {
                    const decoder = new TextDecoder();
                    let finalResponse = '';

                    function readStream() {
                        reader.read().then(({ done, value }) => {
                            if (done) {
                                UI.hideTypingIndicator(); // Stop typing animation
                                UI.setLoading(false);

                                if (finalResponse) {
                                    Chat.addMessage(finalResponse, false, true);
                                }
                                return;
                            }

                            const chunkText = decoder.decode(value, { stream: true });
                            chunkText.split('\n').forEach(line => {
                                if (!line.trim()) return;
                                try {
                                    const json = JSON.parse(line);
                                    if (json.status_update) {
                                        UI.updateStatus(json.status_update);
                                    }
                                    if (json.partial_response) {
                                        finalResponse += json.partial_response;
                                    }
                                } catch {
                                    finalResponse += line;
                                }
                            });

                            readStream();
                        }).catch(error => {
                            console.error('Stream Error:', error);
                            UI.hideTypingIndicator();
                            UI.setLoading(false);
                            UI.displayMessage('Error receiving response.', false);
                        });
                    }

                    readStream();
                })
                .catch(error => {
                    console.error('Fetch Error:', error);
                    UI.hideTypingIndicator();
                    UI.setLoading(false);
                    UI.displayMessage('An error occurred. Please try again.', false);
                });
        },

        resetConversation() {
            const activeId = Storage.getActiveId();
            if (activeId) {
                const conversation = Storage.getConversationById(activeId);
                if (conversation) {
                    conversation.messages = [];
                    Storage.updateConversation(conversation);
                    Conversation.updateTitle(conversation);
                }
            }

            UI.renderActiveConversation();
            UI.renderConversationList();

            fetch('/reset_conversation', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        UI.displayMessage('Conversation has been reset.', false);
                    }
                })
                .catch(error => console.error('Reset Error:', error));
        },

        newChat() {
            const newConv = Conversation.create();
            Storage.updateConversation(newConv);
            Storage.setActiveId(newConv.id);
            UI.renderActiveConversation();
            UI.renderConversationList();
        }
    };

    // -------------------------------
    // EVENT LISTENERS
    // -------------------------------
    elements.agentSelect.addEventListener('change', UI.initializeDropdowns);
    elements.sendBtn.addEventListener('click', Chat.sendMessage);
    elements.userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            Chat.sendMessage();
        }
    });
    elements.resetBtn.addEventListener('click', Chat.resetConversation);
    elements.newChatBtn.addEventListener('click', (e) => {
        e.preventDefault();
        Chat.newChat();
    });

    // -------------------------------
    // INITIALIZATION
    // -------------------------------
    UI.initializeDropdowns();
    UI.renderConversationList();
    UI.renderActiveConversation();
});


// ------------------------------------------------------------
// Function to copy a response to the clipboard
// Clipboard Functionality with Improved Error Handling and Multiple Copy Methods

function copyToClipboard(event) {
    // Prevent default button behavior
    event.preventDefault();

    // Find the closest AI message element
    const aiMessageElement = event.target.closest('.message.ai-message');
    if (!aiMessageElement) {
        showToast("No response to copy!", 'error');
        return;
    }

    // Extract text and HTML content
    const responseContentText = aiMessageElement.innerText.trim();
    const responseContentHTML = aiMessageElement.innerHTML.trim();

    if (!responseContentText) {
        showToast("No response to copy!", 'error');
        return;
    }

    // Prioritize modern Clipboard API
    if (navigator.clipboard && navigator.clipboard.write) {
        try {
            // Create Clipboard Item with both HTML and plain text
            const htmlBlob = new Blob([responseContentHTML], { type: 'text/html' });
            const textBlob = new Blob([responseContentText], { type: 'text/plain' });
            const clipboardItem = new ClipboardItem({
                'text/html': htmlBlob,
                'text/plain': textBlob
            });

            navigator.clipboard.write([clipboardItem])
                .then(() => showToast("Response copied to clipboard!!"))
                .catch(handleCopyError);
        } catch (err) {
            // Fallback to text copy if Clipboard Item creation fails
            fallbackCopyToClipboard(responseContentText);
        }
    } else {
        // Fallback for older browsers
        fallbackCopyToClipboard(responseContentText);
    }
}

function fallbackCopyToClipboard(text) {
    const tempTextArea = document.createElement('textarea');
    tempTextArea.value = text;

    // Styling to make textarea invisible and prevent scroll
    tempTextArea.style.position = 'fixed';
    tempTextArea.style.top = 0;
    tempTextArea.style.left = 0;
    tempTextArea.style.width = '1px';
    tempTextArea.style.height = '1px';
    tempTextArea.style.padding = 0;
    tempTextArea.style.border = 'none';
    tempTextArea.style.outline = 'none';
    tempTextArea.style.boxShadow = 'none';
    tempTextArea.style.background = 'transparent';

    document.body.appendChild(tempTextArea);

    try {
        // Select the text
        tempTextArea.select();
        tempTextArea.setSelectionRange(0, text.length);

        // Execute copy command
        const successful = document.execCommand('copy');

        if (successful) {
            showToast("Response copied to clipboard (Plain Text)!");
        } else {
            showToast("Copying failed, please try again.", 'error');
        }
    } catch (err) {
        showToast("Copying failed, please try again.", 'error');
        console.error("Fallback copy failed:", err);
    } finally {
        // Remove temporary textarea
        document.body.removeChild(tempTextArea);
    }
}

function handleCopyError(err) {
    console.error("Clipboard API error:", err);
    fallbackCopyToClipboard(aiMessageElement.innerText.trim());
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerText = message;

    // Consistent toast styling
    Object.assign(toast.style, {
        position: 'fixed',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: type === 'success' ? '#4CAF50' : '#f44336',
        color: '#fff',
        padding: '10px 20px',
        borderRadius: '5px',
        fontSize: '16px',
        zIndex: '1000',
        opacity: '1',
        transition: 'opacity 0.5s ease-out'
    });

    document.body.appendChild(toast);

    // Auto-remove toast
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 500);
    }, 2000);
}

// Event delegation for copy buttons
document.addEventListener('click', function (event) {
    // Check for both general buttons in AI messages and specific copy buttons
    const copyTrigger = event.target.closest('button');
    if (copyTrigger && (
        copyTrigger.classList.contains('copy-btn') ||
        (copyTrigger.closest('.message.ai-message') && copyTrigger.tagName === 'BUTTON')
    )) {
        copyToClipboard(event);
    }
});

// ------------------------------------------------------------
// Event delegation: Listen for click events on buttons (with a specific class, e.g., "copy-btn")
// that should trigger the copy-to-clipboard functionality
document.addEventListener('click', function (event) {
    if (event.target && event.target.matches('button.copy-btn')) {
        copyToClipboard(event);
    }
});

// ------------------------------------------------------------
// Theme Toggle Functionality
document.addEventListener('DOMContentLoaded', function () {
    const themeToggleButton = document.getElementById('theme-toggle');
    // Note: in the HTML, the icons are swapped (moon and sun) so the variables below are reversed
    const sunIcon = document.getElementById('moon-icon');
    const moonIcon = document.getElementById('sun-icon');

    // Check for a saved theme in localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        enableDarkMode();
    } else {
        enableLightMode();
    }

    // Toggle theme on button click
    themeToggleButton.addEventListener('click', function () {
        if (document.body.classList.contains('dark-mode')) {
            enableLightMode();
            localStorage.setItem('theme', 'light');
        } else {
            enableDarkMode();
            localStorage.setItem('theme', 'dark');
        }
    });

    // Enable dark mode: add a class and swap the icon display
    function enableDarkMode() {
        document.body.classList.add('dark-mode');
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'inline-block';
    }

    // Enable light mode: remove the class and swap the icon display
    function enableLightMode() {
        document.body.classList.remove('dark-mode');
        moonIcon.style.display = 'none';
        sunIcon.style.display = 'inline-block';
    }
});
