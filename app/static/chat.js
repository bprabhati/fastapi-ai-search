const welcome = document.querySelector("#welcome");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt-input");
const sendButton = document.querySelector("#send-button");
const stopButton = document.querySelector("#stop-button");
const newChatButton = document.querySelector("#new-chat-button");
const sessionList = document.querySelector("#session-list");
const statusDot = document.querySelector("#status-dot");
const connectionStatus = document.querySelector("#connection-status");
const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const fileLabel = document.querySelector("#file-label");
const uploadStatus = document.querySelector("#upload-status");
const uploadMessage = document.querySelector("#upload-message");
const documentPicker = document.querySelector("#document-picker");
const documentSelect = document.querySelector("#document-select");
const conversation = document.querySelector(".conversation");
const mainPanel = document.querySelector(".main-panel");
const profileName = document.querySelector("#profile-name");
const logoutButton = document.querySelector("#logout-button");
const authForm = document.querySelector("#auth-form");
const authTitle = document.querySelector("#auth-title");
const authCopy = document.querySelector("#auth-copy");
const authNameField = document.querySelector("#name-field");
const authName = document.querySelector("#auth-name");
const authEmail = document.querySelector("#auth-email");
const authPassword = document.querySelector("#auth-password");
const authSubmit = document.querySelector("#auth-submit");
const authError = document.querySelector("#auth-error");
const authToggleCopy = document.querySelector("#auth-toggle-copy");
const authToggleButton = document.querySelector("#auth-toggle-button");
let accessToken = sessionStorage.getItem("access_token");
let isRegisterMode = false;
let socket = null;
let socketReady = false;
let reconnectTimer = null;
let activeMessage = null;
let activeSources = [];
let currentSessionId = null;

function setAuthenticated(user, token) {
  accessToken = token;
  sessionStorage.setItem("access_token", token);
  mainPanel.classList.add("authenticated");
  profileName.textContent = user.name;
  profileName.hidden = false;
  logoutButton.hidden = false;
  newChatButton.hidden = false;
  authError.textContent = "";
  refreshDocuments();
  refreshSessions();
  connectSocket();
}

function clearAuthentication() {
  accessToken = null;
  sessionStorage.removeItem("access_token");
  disconnectSocket();
  finishAnswer();
  mainPanel.classList.remove("authenticated");
  profileName.textContent = "";
  profileName.hidden = true;
  logoutButton.hidden = true;
  newChatButton.hidden = true;
  currentSessionId = null;
  sessionList.replaceChildren();
  messages.replaceChildren();
  messages.classList.remove("active");
  welcome.classList.remove("hidden");
  documentSelect.replaceChildren(new Option("All documents", ""));
  documentPicker.hidden = true;
  fileInput.value = "";
  fileLabel.textContent = "Add a document";
  uploadStatus.textContent = "PDF or Word file";
  uploadMessage.textContent = "";
}

async function authorizedFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401 && accessToken) clearAuthentication();
  return response;
}

async function refreshDocuments() {
  if (!accessToken) return;
  try {
    const response = await authorizedFetch("/documents");
    const data = await responseData(response);
    if (!response.ok) throw new Error(responseError(data, response.status));

    updateDocumentPicker(data);
  } catch (error) {
    uploadMessage.textContent = `Could not load documents: ${error.message}`;
    uploadMessage.classList.add("error");
  }
}

function setConnected(connected) {
  socketReady = connected;
  statusDot.classList.toggle("offline", !connected);
  connectionStatus.textContent = connected ? "Connected" : "Reconnecting...";
}

function connectSocket() {
  disconnectSocket();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const connection = new WebSocket(`${protocol}://${location.host}/ws/chat`);
  socket = connection;

  connection.addEventListener("open", () => {
    connection.send(JSON.stringify({ type: "auth", token: accessToken }));
  });
  connection.addEventListener("message", (event) => handleSocketMessage(JSON.parse(event.data)));
  connection.addEventListener("close", (event) => {
    if (socket !== connection) return;
    setConnected(false);
    finishAnswer("The connection was lost. Please try again.");
    if (event.code === 4401) clearAuthentication();
    else reconnectTimer = setTimeout(connectSocket, 2000);
  });
}

function disconnectSocket() {
  clearTimeout(reconnectTimer);
  const connection = socket;
  socket = null;
  socketReady = false;
  if (connection) connection.close();
}

function handleSocketMessage(data) {
  if (data.type === "ready") {
    setConnected(true);
  } else if (data.type === "sources") {
    activeSources = data.titles;
    setPendingText("Reading the relevant sections...");
  } else if (data.type === "token") {
    appendToken(data.text);
  } else if (data.type === "done" || data.type === "stopped") {
    finishAnswer();
    if (data.session_id) currentSessionId = data.session_id;
    refreshSessions();
  } else if (data.type === "error") {
    finishAnswer(data.message);
  }
}

function setPendingText(text) {
  if (activeMessage && activeMessage.classList.contains("pending")) {
    activeMessage.querySelector(".message-content").textContent = text;
  }
}

function appendToken(text) {
  if (!activeMessage) return;
  const content = activeMessage.querySelector(".message-content");
  if (activeMessage.classList.contains("pending")) {
    activeMessage.classList.remove("pending");
    content.textContent = "";
  }
  content.append(text);
  conversation.scrollTop = conversation.scrollHeight;
}

function finishAnswer(errorText = null) {
  if (!activeMessage) return;
  const content = activeMessage.querySelector(".message-content");
  if (errorText) {
    content.textContent = errorText;
  } else if (activeMessage.classList.contains("pending")) {
    content.textContent = "Stopped.";
  } else if (activeSources.length > 0) {
    const sources = document.createElement("div");
    sources.className = "message-sources";
    sources.textContent = `Sources: ${activeSources.join(", ")}`;
    content.append(sources);
  }
  activeMessage.classList.remove("pending");
  activeMessage = null;
  activeSources = [];
  sendButton.hidden = false;
  stopButton.hidden = true;
  conversation.scrollTop = conversation.scrollHeight;
  promptInput.focus();
}

async function refreshSessions() {
  if (!accessToken) return;
  try {
    const response = await authorizedFetch("/sessions");
    const data = await responseData(response);
    if (!response.ok) throw new Error(responseError(data, response.status));
    renderSessions(data);
  } catch (error) {
    sessionList.replaceChildren(sessionNotice(`Could not load conversations: ${error.message}`));
  }
}

function sessionNotice(text) {
  const notice = document.createElement("li");
  notice.className = "empty-sessions";
  notice.textContent = text;
  return notice;
}

function renderSessions(sessions) {
  if (sessions.length === 0) {
    sessionList.replaceChildren(sessionNotice("No conversations yet."));
    return;
  }

  sessionList.replaceChildren(
    ...sessions.map((session) => {
      const item = document.createElement("li");
      item.className = `session-item${session.id === currentSessionId ? " active" : ""}`;

      const open = document.createElement("button");
      open.className = "session-open";
      open.type = "button";
      open.textContent = session.title || "Untitled conversation";
      open.title = open.textContent;
      open.addEventListener("click", () => openSession(session.id));

      const remove = document.createElement("button");
      remove.className = "session-delete";
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Delete ${open.textContent}`);
      remove.addEventListener("click", () => deleteSession(session.id));

      item.append(open, remove);
      return item;
    }),
  );
}

async function openSession(sessionId) {
  if (activeMessage) return;
  try {
    const response = await authorizedFetch(`/sessions/${sessionId}/messages`);
    const data = await responseData(response);
    if (!response.ok) throw new Error(responseError(data, response.status));

    currentSessionId = sessionId;
    messages.replaceChildren();
    revealConversation();
    for (const message of data) {
      addMessage("user", message.prompt);
      addMessage("assistant", message.response);
    }
    refreshSessions();
  } catch (error) {
    addMessage("assistant", error.message);
  }
}

async function deleteSession(sessionId) {
  if (activeMessage || !confirm("Delete this conversation?")) return;
  try {
    const response = await authorizedFetch(`/sessions/${sessionId}`, { method: "DELETE" });
    if (!response.ok) throw new Error(responseError({}, response.status));
    if (sessionId === currentSessionId) startNewChat();
    refreshSessions();
  } catch (error) {
    addMessage("assistant", error.message);
  }
}

function startNewChat() {
  if (activeMessage) return;
  currentSessionId = null;
  messages.replaceChildren();
  messages.classList.remove("active");
  welcome.classList.remove("hidden");
  uploadMessage.textContent = "";
  promptInput.focus();
  refreshSessions();
}

function updateDocumentPicker(documents) {
  const selected = documentSelect.value;
  documentSelect.replaceChildren(
    new Option("All documents", ""),
    ...documents.map((doc) => new Option(doc.title, String(doc.id))),
  );
  if (documents.some((doc) => String(doc.id) === selected)) documentSelect.value = selected;
  documentPicker.hidden = documents.length === 0;
}

function revealConversation() {
  welcome.classList.add("hidden");
  messages.classList.add("active");
}

function addMessage(role, text, pending = false) {
  revealConversation();

  const message = document.createElement("div");
  message.className = `message ${role}${pending ? " pending" : ""}`;

  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = role === "assistant" ? "K" : (profileName.textContent.trim()[0] || "You");
  avatar.setAttribute("aria-hidden", "true");

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  message.append(avatar, content);
  messages.append(message);
  conversation.scrollTop = conversation.scrollHeight;
  return message;
}

function responseError(data, status) {
  if (status >= 500) {
    return "Something went wrong on our side. Please try again in a moment.";
  }
  return data.detail || `Request failed (${status}).`;
}

async function responseData(response) {
  const body = await response.text();
  try {
    return body ? JSON.parse(body) : {};
  } catch {
    if (!response.ok) throw new Error(responseError({}, response.status));
    throw new Error("The server returned an invalid response.");
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt || activeMessage) return;
  if (!socketReady) {
    addMessage("assistant", "Not connected yet. Please try again in a moment.");
    return;
  }

  addMessage("user", prompt);
  promptInput.value = "";
  promptInput.style.height = "auto";
  sendButton.hidden = true;
  stopButton.hidden = false;
  activeMessage = addMessage("assistant", "Searching your documents...", true);

  socket.send(JSON.stringify({
    type: "question",
    prompt,
    document_id: documentSelect.value ? Number(documentSelect.value) : null,
    session_id: currentSessionId,
  }));
});

stopButton.addEventListener("click", () => {
  socket.send(JSON.stringify({ type: "stop" }));
  finishAnswer();
});

newChatButton.addEventListener("click", startNewChat);

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 120)}px`;
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt;
    promptInput.focus();
    chatForm.requestSubmit();
  });
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    fileLabel.textContent = file.name;
    uploadForm.requestSubmit();
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  uploadMessage.textContent = "";
  uploadMessage.classList.remove("error");
  uploadStatus.textContent = "Reading your file...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await authorizedFetch("/documents/upload", {
      method: "POST",
      body: formData,
    });
    const data = await responseData(response);
    if (!response.ok) throw new Error(responseError(data, response.status));

    uploadStatus.textContent = "Upload complete";
    uploadMessage.textContent = `${data.title} is ready. Ask a question below.`;
    await refreshDocuments();
    documentSelect.value = String(data.id);
  } catch (error) {
    uploadStatus.textContent = "PDF or Word file";
    uploadMessage.textContent = error.message;
    uploadMessage.classList.add("error");
  }
});

authToggleButton.addEventListener("click", () => {
  isRegisterMode = !isRegisterMode;
  authNameField.hidden = !isRegisterMode;
  authName.required = isRegisterMode;
  authPassword.autocomplete = isRegisterMode ? "new-password" : "current-password";
  authTitle.innerHTML = isRegisterMode ? "Create your <span>account.</span>" : "Welcome <span>back.</span>";
  authCopy.textContent = isRegisterMode
    ? "Create an account to keep your documents private."
    : "Sign in to chat with your private documents.";
  authSubmit.textContent = isRegisterMode ? "Create account" : "Sign in";
  authToggleCopy.textContent = isRegisterMode ? "Already have an account?" : "New here?";
  authToggleButton.textContent = isRegisterMode ? "Sign in" : "Create an account";
  authError.textContent = "";
});

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.textContent = "";
  authSubmit.disabled = true;

  const body = {
    email: authEmail.value.trim(),
    password: authPassword.value,
  };
  if (isRegisterMode) body.name = authName.value.trim();

  try {
    const response = await fetch(isRegisterMode ? "/auth/register" : "/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await responseData(response);
    if (!response.ok) throw new Error(responseError(data, response.status));
    setAuthenticated(data.user, data.access_token);
    authPassword.value = "";
  } catch (error) {
    authError.textContent = error.message;
  } finally {
    authSubmit.disabled = false;
  }
});

logoutButton.addEventListener("click", clearAuthentication);

if (accessToken) {
  authorizedFetch("/auth/me")
    .then(async (response) => {
      const user = await responseData(response);
      if (!response.ok) throw new Error(responseError(user, response.status));
      setAuthenticated(user, accessToken);
    })
    .catch(() => clearAuthentication());
} else {
  profileName.hidden = true;
}
