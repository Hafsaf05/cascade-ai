const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}

export const api = {
  cascades: () => request('/api/risk/cascades'),
  cascade: (id) => request(`/api/risk/cascade/${id}`),
  metrics: () => request('/api/risk/metrics'),
  simulate: (id, transactions) => request('/api/risk/simulate', { method: 'POST', body: JSON.stringify({ cascade_id: id, transactions }) }),
  action: (id, action) => request('/api/risk/action', { method: 'POST', body: JSON.stringify({ cascade_id: id, action, actor: 'investigator' }) }),
}