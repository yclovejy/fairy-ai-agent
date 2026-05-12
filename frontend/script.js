const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const welcomeScreen = document.getElementById("welcome-screen");
const statusBadge = document.getElementById("status-badge");
const newChatBtn = document.getElementById("new-chat-btn");
const clearChatBtn = document.getElementById("clear-chat-btn");
const sidebar = document.querySelector(".sidebar");
const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const suggestionCards = document.querySelectorAll(".suggestion-card");
const modelSelect = document.getElementById("model-select");
const voiceBtn = document.getElementById("voice-btn");
const voiceHint = document.getElementById("voice-hint");

const API_URL = `/chat`;
const PING_URL = `/ping`;
const VOICE_CHAT_URL = `/voice/chat`;

let chatTurns = [];
let isLoading = false;
let currentModel = "default";
let recognition = null;
let isRecording = false;
let speechSynthesisEnabled = "speechSynthesis" in window;

function init() {
    checkBackendStatus();
    
    sendBtn.addEventListener("click", sendMessage);
    
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    userInput.addEventListener("input", autoResize);
    
    newChatBtn.addEventListener("click", newChat);
    clearChatBtn.addEventListener("click", clearChat);
    
    suggestionCards.forEach(card => {
        card.addEventListener("click", () => {
            userInput.value = card.dataset.prompt;
            sendMessage();
        });
    });

    modelSelect.addEventListener("change", (e) => {
        currentModel = e.target.value;
    });

    initSidebar();
    initVoiceRecognition();

    loadChatHistory();
}

function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
}

async function checkBackendStatus() {
    try {
        const response = await fetch(PING_URL);
        if (response.ok) {
            setStatus("online", "在线");
        } else {
            setStatus("offline", "服务异常");
        }
    } catch (error) {
        setStatus("offline", "离线");
    }
}

function setStatus(status, text) {
    const dot = statusBadge.querySelector(".status-dot");
    if (status === "online") {
        dot.className = "status-dot online";
        statusBadge.innerHTML = `<span class="status-dot online"></span>${text}`;
    } else {
        dot.className = "status-dot offline";
        statusBadge.innerHTML = `<span class="status-dot offline"></span>${text}`;
    }
}

async function sendMessage() {
    if (isLoading) return;
    
    const message = userInput.value.trim();
    if (!message) return;

    await submitMessage(message);
}

async function submitMessage(message) {
    if (isLoading) return;

    closeSidebar();
    userInput.value = "";
    autoResize();
    
    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }
    
    addMessage("user", message);

    isLoading = true;
    updateSendButton();
    
    const loadingMessage = addLoadingMessage();
    
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                query: message,
                history: chatTurns,
                model: currentModel
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        
        if (loadingMessage) loadingMessage.remove();
        
        const answer = data.answer || "抱歉，我无法回答这个问题。";
        addMessage("agent", answer, { ttsText: answer });
        chatTurns.push({ user: message, assistant: answer });

        saveChatHistory();
        
        if (chatTurns.length > 25) {
            chatTurns = chatTurns.slice(-25);
        }

    } catch (error) {
        if (loadingMessage) loadingMessage.remove();
        
        let errorMsg = "连接失败，请检查后端服务";
        if (error.message.includes("Failed to fetch")) {
            errorMsg = "无法连接到后端服务，请确保后端运行中";
        }
        
        addMessage("error", errorMsg);
    } finally {
        isLoading = false;
        updateSendButton();
    }
}

async function sendVoiceMessage(transcript) {
    if (isLoading || !transcript) return;

    closeSidebar();
    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }

    addMessage("user", transcript);
    isLoading = true;
    updateSendButton();

    const loadingMessage = addLoadingMessage();

    try {
        const formData = new FormData();
        formData.append("transcript", transcript);
        formData.append("history", JSON.stringify(chatTurns));

        const response = await fetch(VOICE_CHAT_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        if (loadingMessage) loadingMessage.remove();

        const answer = data.answer || "抱歉，我暂时没能处理这段语音。";
        addMessage("agent", answer, { ttsText: data.tts_text || answer });
        chatTurns.push({ user: data.transcript || transcript, assistant: answer });
        saveChatHistory();
    } catch (error) {
        if (loadingMessage) loadingMessage.remove();
        addMessage("error", error.message || "语音请求失败，请稍后重试");
    } finally {
        isLoading = false;
        updateSendButton();
        setVoiceHint("AI生成内容仅供参考，请核实重要信息");
        setVoiceRecording(false);
    }
}

function addMessage(role, content, options = {}) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "✨";
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = formatContent(content);
    
    contentDiv.appendChild(bubble);

    if (role === "agent" && speechSynthesisEnabled) {
        const actions = document.createElement("div");
        actions.className = "message-actions";

        const speakBtn = document.createElement("button");
        speakBtn.className = "speak-btn";
        speakBtn.type = "button";
        speakBtn.textContent = "朗读";
        speakBtn.addEventListener("click", () => {
            speakText(options.ttsText || content);
        });

        actions.appendChild(speakBtn);
        contentDiv.appendChild(actions);
    }
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

function addLoadingMessage() {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message loading";
    
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "✨";
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    
    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.innerHTML = "<span></span><span></span><span></span>";
    
    bubble.appendChild(typing);
    contentDiv.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

function formatContent(text) {
    if (!text) return "";
    
    text = escapeHtml(text);
    
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    text = text.replace(/^\> (.+)$/gm, '<blockquote>$1</blockquote>');
    
    text = text.replace(/\n/g, "<br>");
    
    return text;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function initSidebar() {
    if (!sidebar || !sidebarToggleBtn || !sidebarOverlay) return;

    sidebarToggleBtn.addEventListener("click", toggleSidebar);
    sidebarOverlay.addEventListener("click", closeSidebar);

    window.addEventListener("resize", () => {
        if (window.innerWidth > 768) {
            closeSidebar();
        }
    });
}

function toggleSidebar() {
    if (!sidebar) return;

    const isOpen = sidebar.classList.toggle("open");
    document.body.classList.toggle("sidebar-open", isOpen);
}

function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove("open");
    document.body.classList.remove("sidebar-open");
}

function updateSendButton() {
    sendBtn.disabled = isLoading;
    if (voiceBtn) {
        voiceBtn.disabled = isLoading;
    }
}

function newChat() {
    if (chatTurns.length > 0) {
        if (!confirm("开始新对话将清空当前对话记录，是否继续？")) {
            return;
        }
    }

    closeSidebar();
    chatTurns = [];
    localStorage.removeItem("chatHistory");
    
    const messages = chatMessages.querySelectorAll(".message");
    messages.forEach(msg => msg.remove());
    
    if (welcomeScreen) {
        welcomeScreen.style.display = "flex";
    }
}

function clearChat() {
    if (!confirm("确定要清空所有对话记录吗？")) return;

    closeSidebar();
    chatTurns = [];
    localStorage.removeItem("chatHistory");
    
    const messages = chatMessages.querySelectorAll(".message");
    messages.forEach(msg => msg.remove());
    
    if (welcomeScreen) {
        welcomeScreen.style.display = "flex";
    }
}

function saveChatHistory() {
    try {
        localStorage.setItem("chatHistory", JSON.stringify(chatTurns));
    } catch (e) {
        console.warn("无法保存对话历史:", e);
    }
}

function loadChatHistory() {
    try {
        const saved = localStorage.getItem("chatHistory");
        if (saved) {
            chatTurns = JSON.parse(saved);
            
            if (chatTurns.length > 0 && welcomeScreen) {
                welcomeScreen.style.display = "none";
            }
            
            chatTurns.forEach(t => {
                if (t && typeof t === "object") {
                    if (t.user) addMessage("user", t.user);
                    if (t.assistant) addMessage("agent", t.assistant, { ttsText: t.assistant });
                }
            });
        }
    } catch (e) {
        console.warn("无法加载对话历史:", e);
    }
}

function initVoiceRecognition() {
    if (!voiceBtn) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceBtn.disabled = true;
        setVoiceHint("当前浏览器不支持原生语音识别，可继续使用文本输入。");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "zh-CN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    voiceBtn.addEventListener("click", toggleVoiceInput);

    recognition.onstart = () => {
        setVoiceRecording(true);
        setVoiceHint("正在听你说话，结束后我会自动发送。");
    };

    recognition.onend = () => {
        setVoiceRecording(false);
        if (!isLoading) {
            setVoiceHint("AI生成内容仅供参考，请核实重要信息");
        }
    };

    recognition.onerror = (event) => {
        setVoiceRecording(false);
        setVoiceHint(`语音识别失败：${event.error || "请重试"}`);
    };

    recognition.onresult = async (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0]?.transcript || "")
            .join("")
            .trim();

        if (!transcript) {
            setVoiceHint("没有识别到有效语音，请再试一次。");
            return;
        }

        setVoiceHint("语音识别完成，正在生成回答...");
        await sendVoiceMessage(transcript);
    };
}

function toggleVoiceInput() {
    if (!recognition || isLoading) return;

    if (isRecording) {
        recognition.stop();
        return;
    }

    try {
        recognition.start();
    } catch (error) {
        console.warn("语音识别启动失败:", error);
        setVoiceHint("语音识别启动失败，请检查麦克风权限。");
    }
}

function setVoiceRecording(active) {
    isRecording = active;
    if (!voiceBtn) return;
    voiceBtn.classList.toggle("recording", active);
}

function setVoiceHint(text) {
    if (voiceHint) {
        voiceHint.textContent = text;
    }
}

function speakText(text) {
    if (!speechSynthesisEnabled || !text) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.replace(/\s+/g, " ").trim());
    utterance.lang = "zh-CN";
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
}

window.addEventListener("beforeunload", saveChatHistory);

document.addEventListener("DOMContentLoaded", init);
