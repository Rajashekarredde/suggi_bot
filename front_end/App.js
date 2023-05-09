import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

import botAvatar from './assets/bot.jpg';
import logo from './assets/suggioota.webp'

function App() {
  const [inputText, setInputText] = useState("");
  const [messages, setMessages] = useState([]);
  const chatContainerRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const userMessage = { sender: "user", text: inputText };

    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInputText("");

    try {
      const response = await axios.post("http://localhost:8000/api/predict", {
        data: userMessage.text,
      });
      const botMessage = { sender: "bot", text: response.data };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Error fetching response from API:", error);
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="App">
      <div className="header">
        <h1>Genie Bot</h1>
        <img className="header-image" src={logo} alt="Header Image" />
      </div>
      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((message, index) => (
          <>
            <div key={index} className={`message-container ${message.sender === 'bot' ? 'align-left' : 'align-right'}`}>
              {message.sender === "bot" && (
                <img className="image-bot" src={botAvatar} alt="Bot Avatar" />
              )}
              <div key={index} className={`message ${message.sender}`}>
                {message.text}
              </div>
            </div>
          </>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          className="input-field"
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder='Type your message here (Enter "quit" to exit)'
        />
        {console.log(inputText.length)}
        <button type="submit" disabled={inputText.length == 0 ? "disabled" : ""}>Send</button>
      </form>
      <div className="footer">
        <span>Genie Beta</span>
      </div>
    </div>
  );
}

export default App;
