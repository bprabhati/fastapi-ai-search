const welcome = document.querySelector("#welcome");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt-input");
const sendButton = document.querySelector("#send-button");
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

function setAuthenticated(user, token) {
  accessToken = token;
  sessionStorage.setItem("access_token", token);
  mainPanel.classList.add("authenticated");
  profileName.textContent = user.name;
  profileName.hidden = false;
  logoutButton.hidden = false;
  authError.textContent = "";
  refreshDocuments();
}

function clearAuthentication() {
  accessToken = null;
  sessionStorage.removeItem("access_token");
  mainPanel.classList.remove("authenticated");
  profileName.textContent = "";
  profileName.hidden = true;
  logoutButton.hidden = true;
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

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt || sendButton.disabled) return;

  addMessage("user", prompt);
  promptInput.value = "";
  promptInput.style.height = "auto";
  sendButton.disabled = true;
  const pending = addMessage("assistant", "Thinking...", true);

  try {
    const response = await authorizedFetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        document_id: documentSelect.value ? Number(documentSelect.value) : null,
      }),
    });
    const data = await responseData(response);
    if (!response.ok) throw new Error(responseError(data, response.status));
    pending.querySelector(".message-content").textContent = data.reply;
    pending.classList.remove("pending");
  } catch (error) {
    pending.querySelector(".message-content").textContent = error.message;
    pending.classList.remove("pending");
  } finally {
    sendButton.disabled = false;
    promptInput.focus();
  }
});

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
