"use client";
import { useState, useRef, useEffect } from "react";

interface IMessage {
  type: string;
  text: string;
}

export default function Home() {
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [input, setInput] = useState("");

  const containerRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      const trimmed = input.trim();
      if (!trimmed) return;

      setMessages(prev => [...prev, { type: "user", text: trimmed }]);
      setInput("");
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
      <h1 className="rgb-text">Underground</h1>

      {
        messages?.length ?
          (
            <div
              ref={containerRef}
              className="d-flex flex-column-reverse justify-content-start align-items-center"
              style={{
                height: "70vh",
                width: "800px",
                overflowY: "auto",
                display: "flex",
                flexDirection: "column-reverse"
              }}
            >
              {[...messages].reverse().map((message, k) => (
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
                      backgroundColor: "#151515"
                    }}
                  >
                    {message.text}
                  </div>
                </div>
              ))}
            </div>
          )
          : ""
      }

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
