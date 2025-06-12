"use client";
import { useState, useRef, useEffect, useCallback, ChangeEvent } from "react";

type MessageType = "sys" | "user";
interface IMessage {
  id: string;
  type: MessageType;
  text: string;
}




export default function Home() {
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [input, setInput] = useState("");
  const [maxAmountOfTokens, setMaxAmountOfTokens] = useState<number>(512);

  const containerRef = useRef<HTMLDivElement>(null);
  const evtSourceRef = useRef<EventSource | null>(null);
  const messageQueue = useRef<{ id: string, message: string }[]>([]);

  const handleSysMessage = useCallback((char: string, id: string) => {
    const message = messages.find( m => m.id === id);
    if (!message) {
      console.log("Message not found")
      setMessages(prev => [...prev, { id, type: "sys", text: char }]);
      return;
    }
    console.log("Updating message")
    setMessages(prevMessages =>
      prevMessages.map(message =>
        message.id === id
          ? { ...message, text: message.text + char }
          : message
      )
    );
  }, [messages]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (messageQueue.current.length > 0) {
        const { id, message } = messageQueue.current.shift()!;
        handleSysMessage(message, id);
      }
    }, 30);
  
    return () => clearInterval(interval);
  }, [handleSysMessage]);

  const submit = useCallback((input: string) => {
    if (evtSourceRef.current) {
      evtSourceRef.current.close();
    }

    const messageId = crypto.randomUUID();
    const encoded = encodeURIComponent(input);
    const evtSource = new EventSource(`https://fb9c-2001-1284-f508-4a6c-1cc-8f-6871-d0bc.ngrok-free.app/sse?prompt=${encoded}&max_tokens=${maxAmountOfTokens}&id=${messageId}`);

    evtSource.onmessage = function(event) {
      if (event.data === "[DONE]") {
        evtSource.close();
        return;
      }
      console.log('New message:', event.data);
      messageQueue.current.push({ id: messageId, message: event.data });
    };

    evtSource.onerror = (err) => {
      console.error('EventSource failed:', err);
      evtSource.close();
    };

    evtSourceRef.current = evtSource;

  }, [maxAmountOfTokens]);

  const handleMaxTokensChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setMaxAmountOfTokens(Number(event.target.value));
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      const trimmed = input.trim();
      if (!trimmed) return;

      setMessages(prev => [...prev, { id: crypto.randomUUID(), type: "user", text: trimmed }]);
      setInput("");
      submit(trimmed);
    }
  };

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div
      className="d-flex flex-column justify-content-center align-items-center"
      style={{
        width: "100vw",
        height: "100vh",
        backgroundColor: "#0a0a0a",
        color: "#fff"
      }}
    >
      <h1 className="rgb-text">Midnight</h1>

      {
        messages?.length ?
          (
            <div
              ref={containerRef}
              className="d-flex flex-column-reverse justify-content-start align-items-center"
              style={{
                height: "50vh",
                width: "800px",
                overflowY: "auto",
                display: "flex",
                flexDirection: "column-reverse"
              }}
            >
              {[...messages].reverse().map((message, k) => {
                return (
                  <div
                    key={k}
                    className={`d-flex flex-column justify-content-center align-items-${message.type === "user" ? "end" : "start"} mb-3`}
                    style={{ width: "100%" }}
                  >
                    <div
                      className="d-flex flex-column p-2"
                      style={{
                        width: "400px",
                        borderRadius: "8px",
                        backgroundColor: message?.type == 'sys' ? "transparent" : "#151515"
                      }}
                    >
                      {message.text}
                    </div>
                  </div>
                )
                
              })}
            </div>
          )
          : ""
      }

      <div className="d-flex flex-column justify-content-start align-items-start mt-3" style={{ width: "800px" }}>
        <label
          style={{
            fontSize: "0.9em",
            color: "#cecece",
            marginBottom: "2px"
          }}
        >Max amount of tokens ( response )</label>
        <input
          type="text"
          className="form-control"
          onChange={handleMaxTokensChange}
          value={maxAmountOfTokens}
          style={{
            height: "35px",
            backgroundColor: "transparent",
            color: "#fff",
            border: "1px solid #202020"
          }}
        />
      </div>


      <div className="d-flex justify-content-center align-items-center mt-3">
        <textarea
          placeholder="Type something..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          style={{
            backgroundColor: "#101010",
            border: "none",
            width: "800px",
            height: "100px",
            borderRadius: "4px",
            outline: "none",
            padding: "10px 20px",
            resize: "none",
            color: "#fff"
          }}
        />
      </div>
    </div>
  );
}
