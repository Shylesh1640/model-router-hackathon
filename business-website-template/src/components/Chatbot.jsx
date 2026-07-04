import React, { useState, useRef, useEffect } from 'react'

const dummyResponses = [
  "Thanks for reaching out! Our team typically responds within 24 hours.",
  "That's a great question! You can learn more on our Features page.",
  "We'd love to help you with that. Could you share a bit more detail?",
  "Great news! We have a solution for exactly that use case.",
  "Feel free to browse our website for more information in the meantime!",
]

const keywordResponses = {
  pricing: "Our pricing is competitive and tailored to your needs. Contact our sales team for a custom quote!",
  price: "Our pricing is competitive and tailored to your needs. Contact our sales team for a custom quote!",
  cost: "Our pricing is competitive and tailored to your needs. Contact our sales team for a custom quote!",
  hello: "Hello! Welcome to HooBank. How can I assist you today?",
  hi: "Hi there! Welcome to HooBank. How can I assist you today?",
  help: "I'm here to help! You can ask me about our features, pricing, or how to get started.",
  feature: "We offer rewards, 100% secure transactions, and balance transfer options. Check out our Features section!",
  contact: "You can reach us through the contact form or connect with us on social media.",
  thanks: "You're welcome! Feel free to ask if you have any more questions.",
  thank: "You're welcome! Feel free to ask if you have any more questions.",
}

const Chatbot = () => {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hi! Welcome to HooBank. How can I help you today?' },
  ])
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userText = input.trim()
    setMessages((prev) => [...prev, { from: 'user', text: userText }])
    setInput('')

    setTimeout(() => {
      const lower = userText.toLowerCase()
      let reply = null

      for (const [keyword, response] of Object.entries(keywordResponses)) {
        if (lower.includes(keyword)) {
          reply = response
          break
        }
      }

      if (!reply) {
        reply = dummyResponses[Math.floor(Math.random() * dummyResponses.length)]
      }

      setMessages((prev) => [...prev, { from: 'bot', text: reply }])
    }, 600)
  }

  return (
    <div className='fixed bottom-6 right-6 z-50 flex flex-col items-end'>
      {open && (
        <div className='mb-4 w-[340px] sm:w-[380px] bg-primary border border-[#3b3b4a] rounded-2xl shadow-2xl overflow-hidden animate-slide-up'>
          <div className='bg-black-gradient-2 px-5 py-4 flex items-center justify-between'>
            <div className='flex items-center gap-3'>
              <div className='w-9 h-9 rounded-full bg-blue-gradient flex items-center justify-center'>
                <svg className='w-5 h-5 text-primary' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                  <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z' />
                </svg>
              </div>
              <div>
                <p className='font-poppins font-semibold text-white text-[15px]'>Chat Support</p>
                <p className='font-poppins text-[11px] text-dimWhite'>Typically replies instantly</p>
              </div>
            </div>
            <button onClick={() => setOpen(false)} className='text-dimWhite hover:text-white transition-colors'>
              <svg className='w-5 h-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M6 18L18 6M6 6l12 12' />
              </svg>
            </button>
          </div>

          <div className='h-[320px] overflow-y-auto px-4 py-4 space-y-3 scrollbar-thin'>
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.from === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] px-4 py-2.5 rounded-2xl font-poppins text-[14px] leading-relaxed ${
                  msg.from === 'user'
                    ? 'bg-blue-gradient text-primary rounded-br-md'
                    : 'bg-[#1c1c2e] text-dimWhite rounded-bl-md'
                }`}>
                  {msg.text}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} className='border-t border-[#3b3b4a] px-4 py-3 flex items-center gap-3'>
            <input
              type='text'
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder='Type your message...'
              className='flex-1 bg-transparent border-none outline-none font-poppins text-[14px] text-white placeholder-dimWhite'
            />
            <button
              type='submit'
              disabled={!input.trim()}
              className='w-9 h-9 rounded-full bg-blue-gradient flex items-center justify-center disabled:opacity-40 transition-opacity'
            >
              <svg className='w-4 h-4 text-primary' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M5 12h14M12 5l7 7-7 7' />
              </svg>
            </button>
          </form>
        </div>
      )}

      <button
        onClick={() => setOpen((prev) => !prev)}
        className='w-14 h-14 rounded-full bg-blue-gradient shadow-lg flex items-center justify-center hover:scale-105 transition-transform'
      >
        {open ? (
          <svg className='w-6 h-6 text-primary' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
            <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M6 18L18 6M6 6l12 12' />
          </svg>
        ) : (
          <svg className='w-6 h-6 text-primary' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
            <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z' />
          </svg>
        )}
      </button>
    </div>
  )
}

export default Chatbot
