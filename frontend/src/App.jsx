import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'
import useWebSocket from 'react-use-websocket'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [generatedText, setGeneratedText] = useState([])
  const [metrics, setMetrics] = useState({
    throughput: [],
    latency: [],
    acceptanceRate: 0,
    totalTokens: 0
  })
  const [isGenerating, setIsGenerating] = useState(false)

  const { lastMessage } = useWebSocket(`${API_URL.replace('http', 'ws')}/ws/metrics`)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsGenerating(true)
    
    try {
      const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_length: 128 })
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const data = JSON.parse(decoder.decode(value))
        setGeneratedText(prev => [...prev, {
          token: data.token,
          accepted: data.accepted,
          speculative: data.speculative
        }])

        setMetrics(prev => ({
          ...prev,
          throughput: [...prev.throughput.slice(-19), data.tokens_per_sec],
          latency: [...prev.latency.slice(-19), data.latency],
          acceptanceRate: data.acceptance_rate,
          totalTokens: data.total_tokens
        }))
      }
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Speculative Decoding Visualizer
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full p-4 bg-gray-800 rounded-lg border-2 border-cyan-500/20 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/30 transition-all"
            placeholder="Enter your prompt..."
            rows={3}
            disabled={isGenerating}
          />
          <button
            type="submit"
            disabled={isGenerating}
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : 'Generate'}
          </button>
        </form>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <TokenStream tokens={generatedText} />
          <MetricsDashboard metrics={metrics} />
        </div>
      </div>
    </div>
  )
}

const TokenStream = ({ tokens }) => (
  <div className="p-6 bg-gray-800 rounded-xl border border-cyan-500/20">
    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Token Stream</h2>
    <div className="flex flex-wrap gap-2">
      <AnimatePresence>
        {tokens.map((token, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`px-2 py-1 rounded-md text-sm font-mono ${token.speculative ? 'bg-yellow-500/20 text-yellow-300 animate-pulse' : 'bg-cyan-500/20 text-cyan-300'}`}
          >
            {token.token}
          </motion.span>
        ))}
      </AnimatePresence>
    </div>
  </div>
)

const MetricsDashboard = ({ metrics }) => (
  <div className="space-y-6">
    <div className="grid grid-cols-2 gap-4">
      <MetricCard label="Tokens/sec" value={metrics.throughput.slice(-1)[0]?.toFixed(1) || 0} />
      <MetricCard label="Acceptance Rate" value={`${(metrics.acceptanceRate * 100).toFixed(1)}%`} />
    </div>

    <ChartContainer title="Throughput Over Time">
      <LineChart data={metrics.throughput.map((v, i) => ({ time: i, value: v }))}>
        <Line type="monotone" dataKey="value" stroke="#22d3ee" strokeWidth={2} dot={false} />
        <XAxis dataKey="time" stroke="#4b5563" />
        <YAxis stroke="#4b5563" />
        <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
      </LineChart>
    </ChartContainer>
  </div>
)

const MetricCard = ({ label, value }) => (
  <div className="p-4 bg-gray-800 rounded-lg border border-cyan-500/20">
    <div className="text-sm text-cyan-400">{label}</div>
    <div className="text-2xl font-bold mt-2">{value}</div>
  </div>
)

const ChartContainer = ({ title, children }) => (
  <div className="p-4 bg-gray-800 rounded-xl border border-cyan-500/20">
    <h3 className="text-lg font-semibold mb-4 text-cyan-400">{title}</h3>
    <div className="h-48">
      {children}
    </div>
  </div>
)