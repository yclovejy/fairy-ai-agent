const sendBtn = document.getElementById("send-btn");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const welcomeScreen = document.getElementById("welcome-screen");
const statusBadge = document.getElementById("status-badge");
const newChatBtn = document.getElementById("new-chat-btn");
const clearChatBtn = document.getElementById("clear-chat-btn");
const agentEye = document.getElementById("agent-eye");
const quickPrompts = document.getElementById("quick-prompts");
const chatHistoryEl = document.getElementById("chat-history");
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
const CONVERSATIONS_KEY = "fairyConversations";
const ACTIVE_CONVERSATION_KEY = "fairyActiveConversationId";
const LEGACY_CHAT_HISTORY_KEY = "chatHistory";

let chatTurns = [];
let conversations = [];
let activeConversationId = null;
let isLoading = false;
let currentModel = "default";
let recognition = null;
let isRecording = false;
let speechSynthesisEnabled = "speechSynthesis" in window;

function init() {
    checkBackendStatus();

    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            sendMessage();
        });
    } else {
        sendBtn.addEventListener("click", sendMessage);
    }
    
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

    if (modelSelect) {
        modelSelect.addEventListener("change", (e) => {
            currentModel = e.target.value;
            const conversation = getActiveConversation();
            if (conversation) {
                conversation.model = currentModel;
                persistConversations();
            }
        });
    }

    initSidebar();
    initVoiceRecognition();

    loadChatHistory();
}

function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
}

function createConversationId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
    }
    return `fairy-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function normalizeTurns(turns) {
    if (!Array.isArray(turns)) return [];
    return turns
        .filter(turn => turn && typeof turn === "object" && turn.user && turn.assistant)
        .map(turn => ({
            user: String(turn.user),
            assistant: String(turn.assistant),
        }));
}

function createConversation(firstMessage = "") {
    const now = new Date().toISOString();
    return {
        id: createConversationId(),
        title: buildConversationTitle(firstMessage),
        turns: [],
        createdAt: now,
        updatedAt: now,
        model: currentModel,
    };
}

function normalizeConversation(raw) {
    if (!raw || typeof raw !== "object") return null;
    const turns = normalizeTurns(raw.turns);
    const createdAt = raw.createdAt || raw.updatedAt || new Date().toISOString();
    const updatedAt = raw.updatedAt || createdAt;
    const firstUserMessage = turns[0]?.user || "";
    return {
        id: raw.id || createConversationId(),
        title: raw.title || buildConversationTitle(firstUserMessage),
        turns,
        createdAt,
        updatedAt,
        model: raw.model || "default",
    };
}

function buildConversationTitle(message) {
    const cleaned = (message || "").replace(/\s+/g, " ").trim();
    if (!cleaned) return "新对话";
    return cleaned.length > 22 ? `${cleaned.slice(0, 22)}...` : cleaned;
}

function getActiveConversation() {
    return conversations.find(conversation => conversation.id === activeConversationId) || null;
}

function ensureActiveConversation(firstMessage = "") {
    let conversation = getActiveConversation();
    if (conversation) return conversation;

    conversation = createConversation(firstMessage);
    activeConversationId = conversation.id;
    conversations.unshift(conversation);
    return conversation;
}

function persistConversations() {
    try {
        const visibleConversations = conversations.filter(conversation => conversation.turns.length > 0);
        localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(visibleConversations));
        localStorage.removeItem(LEGACY_CHAT_HISTORY_KEY);
        if (activeConversationId) {
            localStorage.setItem(ACTIVE_CONVERSATION_KEY, activeConversationId);
        } else {
            localStorage.removeItem(ACTIVE_CONVERSATION_KEY);
        }
    } catch (e) {
        console.warn("无法保存对话列表:", e);
    }
}

function saveActiveConversation(options = {}) {
    const touch = Boolean(options.touch);
    const conversation = getActiveConversation();
    if (!conversation) {
        persistConversations();
        renderConversationHistory();
        return;
    }

    conversation.turns = normalizeTurns(chatTurns);
    conversation.model = currentModel;
    if (touch) {
        conversation.updatedAt = new Date().toISOString();
    }
    if (conversation.turns[0]?.user) {
        conversation.title = buildConversationTitle(conversation.turns[0].user);
    }

    conversations = [
        conversation,
        ...conversations.filter(item => item.id !== conversation.id),
    ];
    persistConversations();
    renderConversationHistory();
}

function sameCalendarDay(a, b) {
    return a.getFullYear() === b.getFullYear()
        && a.getMonth() === b.getMonth()
        && a.getDate() === b.getDate();
}

function getConversationGroupLabel(dateValue) {
    const date = new Date(dateValue);
    const now = new Date();
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);

    if (sameCalendarDay(date, now)) return "今天";
    if (sameCalendarDay(date, yesterday)) return "昨天";
    return "更早";
}

function renderConversationHistory() {
    if (!chatHistoryEl) return;

    chatHistoryEl.innerHTML = "";
    const savedConversations = conversations
        .filter(conversation => conversation.turns.length > 0)
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

    if (!savedConversations.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = "暂无历史对话";
        chatHistoryEl.appendChild(empty);
        return;
    }

    const groups = new Map();
    savedConversations.forEach(conversation => {
        const label = getConversationGroupLabel(conversation.updatedAt);
        if (!groups.has(label)) groups.set(label, []);
        groups.get(label).push(conversation);
    });

    ["今天", "昨天", "更早"].forEach(label => {
        const group = groups.get(label);
        if (!group || !group.length) return;

        const section = document.createElement("section");
        section.className = "history-group";

        const heading = document.createElement("p");
        heading.className = "history-label";
        heading.textContent = label;
        section.appendChild(heading);

        group.forEach(conversation => {
            const item = document.createElement("button");
            item.className = "history-item";
            item.type = "button";
            item.dataset.conversationId = conversation.id;
            item.classList.toggle("active", conversation.id === activeConversationId);
            item.innerHTML = `
                <span>${escapeHtml(conversation.title)}</span>
                <small>${conversation.turns.length}轮</small>
            `;
            item.addEventListener("click", () => loadConversation(conversation.id));
            section.appendChild(item);
        });

        chatHistoryEl.appendChild(section);
    });
}

function renderCurrentConversation() {
    const messages = chatMessages.querySelectorAll(".message");
    messages.forEach(msg => msg.remove());

    if (welcomeScreen) {
        welcomeScreen.style.display = chatTurns.length > 0 ? "none" : "flex";
    }

    chatTurns.forEach(turn => {
        addMessage("user", turn.user);
        addMessage("agent", turn.assistant, { ttsText: turn.assistant });
    });

    updateChatSurfaceState();
    scrollToBottom();
}

function loadConversation(conversationId) {
    if (isLoading) return;

    const conversation = conversations.find(item => item.id === conversationId);
    if (!conversation) return;

    closeSidebar();
    activeConversationId = conversation.id;
    currentModel = conversation.model || currentModel;
    if (modelSelect) {
        modelSelect.value = currentModel;
    }
    chatTurns = normalizeTurns(conversation.turns);
    persistConversations();
    renderConversationHistory();
    renderCurrentConversation();
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
    setAgentThinking(true);
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
        const conversation = ensureActiveConversation(message);
        chatTurns = conversation.turns;
        chatTurns.push({ user: message, assistant: answer });
        
        if (chatTurns.length > 25) {
            chatTurns = chatTurns.slice(-25);
        }
        saveActiveConversation({ touch: true });

    } catch (error) {
        if (loadingMessage) loadingMessage.remove();
        
        let errorMsg = "连接失败，请检查后端服务";
        if (error.message.includes("Failed to fetch")) {
            errorMsg = "无法连接到后端服务，请确保后端运行中";
        }
        
        addMessage("error", errorMsg);
    } finally {
        isLoading = false;
        setAgentThinking(false);
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
    setAgentThinking(true);
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
        const conversation = ensureActiveConversation(data.transcript || transcript);
        chatTurns = conversation.turns;
        chatTurns.push({ user: data.transcript || transcript, assistant: answer });
        if (chatTurns.length > 25) {
            chatTurns = chatTurns.slice(-25);
        }
        saveActiveConversation({ touch: true });
    } catch (error) {
        if (loadingMessage) loadingMessage.remove();
        addMessage("error", error.message || "语音请求失败，请稍后重试");
    } finally {
        isLoading = false;
        setAgentThinking(false);
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
    if (role === "user") {
        avatar.classList.add("user-avatar");
    } else {
        avatar.classList.add("agent-avatar");
        avatar.appendChild(createAgentEyeAvatar());
    }
    
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
    updateChatSurfaceState();
    scrollToBottom();
    
    return messageDiv;
}

function addLoadingMessage() {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message loading";
    
    const avatar = document.createElement("div");
    avatar.className = "message-avatar agent-avatar";
    avatar.appendChild(createAgentEyeAvatar(true));
    
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
    updateChatSurfaceState();
    scrollToBottom();
    
    return messageDiv;
}

function createAgentEyeAvatar(thinking = false) {
    const eye = document.createElement("div");
    eye.className = `agent-eye agent-eye-mini${thinking ? " thinking" : ""}`;
    eye.setAttribute("aria-hidden", "true");
    eye.innerHTML = `
        <div class="eye-shell">
            <div class="eye-white">
                <div class="iris"></div>
                <div class="catchlight"></div>
            </div>
            <div class="eyelid"></div>
            <div class="thinking-line"></div>
        </div>
    `;
    return eye;
}

function setAgentThinking(active) {
    document.body.classList.toggle("agent-thinking", active);
    if (agentEye) {
        agentEye.classList.toggle("thinking", active);
    }
}

function updateChatSurfaceState() {
    const hasMessages = chatMessages.querySelectorAll(".message:not(.loading)").length > 0;
    const main = document.querySelector(".chat-main");
    if (main) {
        main.classList.toggle("has-messages", hasMessages);
    }
    if (quickPrompts) {
        quickPrompts.hidden = !hasMessages;
    }
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
    if (isLoading) return;

    closeSidebar();
    saveActiveConversation();
    activeConversationId = null;
    chatTurns = [];
    persistConversations();
    renderConversationHistory();
    setAgentThinking(false);
    
    const messages = chatMessages.querySelectorAll(".message");
    messages.forEach(msg => msg.remove());
    
    if (welcomeScreen) {
        welcomeScreen.style.display = "flex";
    }
    updateChatSurfaceState();
}

function clearChat() {
    if (isLoading) return;
    if (!confirm("确定要删除当前对话吗？")) return;

    closeSidebar();
    if (activeConversationId) {
        conversations = conversations.filter(conversation => conversation.id !== activeConversationId);
    }
    activeConversationId = null;
    chatTurns = [];
    persistConversations();
    renderConversationHistory();
    setAgentThinking(false);
    
    const messages = chatMessages.querySelectorAll(".message");
    messages.forEach(msg => msg.remove());
    
    if (welcomeScreen) {
        welcomeScreen.style.display = "flex";
    }
    updateChatSurfaceState();
}

function saveChatHistory() {
    saveActiveConversation();
}

function loadChatHistory() {
    try {
        const saved = localStorage.getItem(CONVERSATIONS_KEY);
        const parsed = saved ? JSON.parse(saved) : [];
        conversations = Array.isArray(parsed)
            ? parsed.map(normalizeConversation).filter(Boolean)
            : [];

        const legacy = localStorage.getItem(LEGACY_CHAT_HISTORY_KEY);
        if (!conversations.length && legacy) {
            const legacyTurns = normalizeTurns(JSON.parse(legacy));
            if (legacyTurns.length) {
                const migrated = createConversation(legacyTurns[0].user);
                migrated.turns = legacyTurns;
                migrated.createdAt = new Date().toISOString();
                migrated.updatedAt = migrated.createdAt;
                conversations = [migrated];
                activeConversationId = migrated.id;
            }
        } else {
            activeConversationId = localStorage.getItem(ACTIVE_CONVERSATION_KEY);
        }

        if (!conversations.some(conversation => conversation.id === activeConversationId)) {
            activeConversationId = conversations[0]?.id || null;
        }

        const activeConversation = getActiveConversation();
        chatTurns = activeConversation ? normalizeTurns(activeConversation.turns) : [];
        if (activeConversation?.model) {
            currentModel = activeConversation.model;
            if (modelSelect) {
                modelSelect.value = currentModel;
            }
        }
    } catch (e) {
        console.warn("无法加载对话列表:", e);
        conversations = [];
        activeConversationId = null;
        chatTurns = [];
    }

    persistConversations();
    renderConversationHistory();
    renderCurrentConversation();
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
