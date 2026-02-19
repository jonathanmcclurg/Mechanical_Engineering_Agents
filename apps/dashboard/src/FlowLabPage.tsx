import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type CaseFormState = {
  failure_type: string
  failure_description: string
  part_number: string
  serial_number: string
  test_name: string
  test_value: string
  spec_lower: string
  spec_upper: string
}

type StageSession = {
  session_id: string
  status: string
  created_at: string
  updated_at: string
  completed_stages: string[]
  next_stage?: string | null
  last_error?: string | null
  workflow_log: Array<Record<string, unknown>>
  agent_logs: Record<string, Array<Record<string, unknown>>>
  outputs: Record<string, unknown>
}

type StageRunResponse = {
  session: StageSession
  stage_result: {
    stage: string
    output: Record<string, unknown>
    workflow_log: Array<Record<string, unknown>>
    agent_logs: Record<string, Array<Record<string, unknown>>>
  }
}

const initialFormState: CaseFormState = {
  failure_type: 'leak_test_fail',
  failure_description: 'Unit failed helium leak test. Leak rate exceeded upper limit.',
  part_number: 'HYD-VALVE-200',
  serial_number: '',
  test_name: 'Helium_Leak_Test',
  test_value: '2.3e-6',
  spec_lower: '0',
  spec_upper: '1e-6',
}

const formatJson = (value: unknown) => JSON.stringify(value, null, 2)

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return null
}

export default function FlowLabPage() {
  const [formState, setFormState] = useState<CaseFormState>(initialFormState)
  const [stages, setStages] = useState<string[]>([])
  const [session, setSession] = useState<StageSession | null>(null)
  const [selectedStage, setSelectedStage] = useState<string>('')
  const [lastStageOutput, setLastStageOutput] = useState<Record<string, unknown> | null>(null)
  const [errorMessage, setErrorMessage] = useState<string>('')
  const [statusMessage, setStatusMessage] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)

  useEffect(() => {
    const loadStages = async () => {
      try {
        const response = await fetch(`${API_BASE}/flow/stages`)
        if (!response.ok) {
          throw new Error('Failed to load stage list.')
        }
        const data = await response.json()
        const stageList = data.stages || []
        setStages(stageList)
        if (stageList.length > 0) {
          setSelectedStage(stageList[0])
        }
      } catch (err) {
        setErrorMessage((err as Error).message)
      }
    }
    loadStages()
  }, [])

  const outputForSelectedStage = useMemo(() => {
    if (!session || !selectedStage) {
      return null
    }
    const value = session.outputs[selectedStage]
    return asRecord(value)
  }, [session, selectedStage])

  const ragCitationCount = useMemo(() => {
    const pgOutput = asRecord(session?.outputs?.product_guide)
    const citations = pgOutput?.citations_used
    return Array.isArray(citations) ? citations.length : 0
  }, [session])

  const handleChange = (field: keyof CaseFormState, value: string) => {
    setFormState((prev) => ({ ...prev, [field]: value }))
  }

  const createSession = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setLoading(true)
    setErrorMessage('')
    setStatusMessage('')
    setLastStageOutput(null)

    const payload: Record<string, unknown> = {
      failure_type: formState.failure_type.trim(),
      failure_description: formState.failure_description.trim(),
      part_number: formState.part_number.trim(),
    }

    ;['serial_number', 'test_name'].forEach((key) => {
      const value = formState[key as keyof CaseFormState].trim()
      if (value) {
        payload[key] = value
      }
    })

    ;['test_value', 'spec_lower', 'spec_upper'].forEach((key) => {
      const value = formState[key as keyof CaseFormState].trim()
      if (value) {
        payload[key] = Number(value)
      }
    })

    try {
      const response = await fetch(`${API_BASE}/flow/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to create flow session.')
      }
      const data = await response.json()
      setSession(data)
      setStatusMessage(`Flow session ready: ${data.session_id}`)
    } catch (err) {
      setErrorMessage((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const runStage = async (stage?: string) => {
    if (!session) {
      return
    }
    setLoading(true)
    setErrorMessage('')
    setStatusMessage('')

    try {
      const response = await fetch(`${API_BASE}/flow/sessions/${session.session_id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: stage || selectedStage }),
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Stage run failed.')
      }
      const data: StageRunResponse = await response.json()
      setSession(data.session)
      setLastStageOutput(data.stage_result.output)
      setSelectedStage(data.stage_result.stage)
      setStatusMessage(`Completed stage: ${data.stage_result.stage}`)
    } catch (err) {
      setErrorMessage((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dashboard flow-lab">
      <header className="dashboard__header">
        <div>
          <p className="eyebrow">Agentic Flow Lab</p>
          <h1>Section-by-Section Debugger</h1>
          <p className="subtitle">
            Run each agent stage independently, inspect outputs, and focus on RAG behavior before continuing.
          </p>
        </div>
        <div className="header-actions">
          <a className="btn btn--secondary" href="#/">
            Open dashboard
          </a>
          <button
            className="btn btn--secondary"
            disabled={!session || loading || !session.next_stage}
            onClick={() => runStage(session?.next_stage ?? undefined)}
          >
            {loading ? 'Running…' : `Run next${session?.next_stage ? ` (${session.next_stage})` : ''}`}
          </button>
        </div>
      </header>

      {errorMessage && <div className="banner banner--error">{errorMessage}</div>}
      {statusMessage && <div className="banner banner--success">{statusMessage}</div>}

      <main className="dashboard__content">
        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Session setup</h2>
          </div>
          <div className="panel__body">
            <form className="form" onSubmit={createSession}>
              <div className="form-grid">
                <label>
                  Failure type
                  <input
                    type="text"
                    required
                    value={formState.failure_type}
                    onChange={(event) => handleChange('failure_type', event.target.value)}
                  />
                </label>
                <label>
                  Part number
                  <input
                    type="text"
                    required
                    value={formState.part_number}
                    onChange={(event) => handleChange('part_number', event.target.value)}
                  />
                </label>
                <label className="form-grid__full">
                  Failure description
                  <textarea
                    required
                    rows={3}
                    value={formState.failure_description}
                    onChange={(event) => handleChange('failure_description', event.target.value)}
                  />
                </label>
                <label>
                  Serial number
                  <input
                    type="text"
                    value={formState.serial_number}
                    onChange={(event) => handleChange('serial_number', event.target.value)}
                  />
                </label>
                <label>
                  Test name
                  <input
                    type="text"
                    value={formState.test_name}
                    onChange={(event) => handleChange('test_name', event.target.value)}
                  />
                </label>
                <label>
                  Test value
                  <input
                    type="text"
                    value={formState.test_value}
                    onChange={(event) => handleChange('test_value', event.target.value)}
                  />
                </label>
                <label>
                  Spec lower
                  <input
                    type="text"
                    value={formState.spec_lower}
                    onChange={(event) => handleChange('spec_lower', event.target.value)}
                  />
                </label>
                <label>
                  Spec upper
                  <input
                    type="text"
                    value={formState.spec_upper}
                    onChange={(event) => handleChange('spec_upper', event.target.value)}
                  />
                </label>
              </div>
              <div className="form-actions">
                <button type="submit" className="btn" disabled={loading}>
                  {loading ? 'Creating…' : 'Create flow session'}
                </button>
              </div>
            </form>
          </div>
        </section>

        <section className="panel">
          <div className="panel__header">
            <h2>Stages</h2>
          </div>
          <div className="panel__body">
            {!session ? (
              <p className="muted">Create a session to run stages.</p>
            ) : (
              <ul className="list">
                {stages.map((stage) => {
                  const completed = session.completed_stages.includes(stage)
                  return (
                    <li
                      key={stage}
                      className={`list-item ${selectedStage === stage ? 'is-active' : ''}`}
                      onClick={() => setSelectedStage(stage)}
                    >
                      <div>
                        <p className="list-title">{stage}</p>
                        <p className="list-subtitle">{completed ? 'completed' : 'pending'}</p>
                      </div>
                      <button
                        type="button"
                        className="btn btn--secondary"
                        onClick={(event) => {
                          event.stopPropagation()
                          runStage(stage)
                        }}
                        disabled={loading}
                      >
                        Run
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel__header">
            <h2>RAG quick view</h2>
          </div>
          <div className="panel__body">
            <div className="detail-grid">
              <div>
                <p className="muted">Session ID</p>
                <p>{session?.session_id || '—'}</p>
              </div>
              <div>
                <p className="muted">Status</p>
                <p>{session?.status || '—'}</p>
              </div>
              <div>
                <p className="muted">Completed stages</p>
                <p>{session?.completed_stages.length || 0}</p>
              </div>
              <div>
                <p className="muted">Product-guide citations</p>
                <p>{ragCitationCount}</p>
              </div>
            </div>
            <p className="muted">
              Focus on `product_guide` output and citations before advancing to `research` and beyond.
            </p>
          </div>
        </section>

        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Selected stage output</h2>
          </div>
          <div className="panel__body trace-panel">
            <pre>
              <code>{formatJson(outputForSelectedStage || lastStageOutput || { message: 'No output yet.' })}</code>
            </pre>
          </div>
        </section>

        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Workflow log</h2>
          </div>
          <div className="panel__body trace-panel">
            <pre>
              <code>{formatJson(session?.workflow_log || [])}</code>
            </pre>
          </div>
        </section>
      </main>
    </div>
  )
}
