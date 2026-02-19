import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type Metrics = {
  cases: {
    total: number
    completed: number
    failed: number
    pending: number
  }
  feedback: {
    total: number
    useful_reports: number
  }
  timestamp: string
}

type FailureCase = {
  case_id: string
  status: string
  submitted_at: string
  report_id?: string
  error?: string
  data?: Record<string, unknown>
  trace_md_path?: string
}

type ReportItem = {
  report_id: string
  case_id: string
  created_at: string
  report?: Record<string, unknown>
  trace_md_path?: string
}

type ReportDetail = {
  report_id: string
  case_id: string
  status: string
  report?: Record<string, unknown>
  error?: string
}

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

type PanelKey =
  | 'cases'
  | 'reports'
  | 'caseDetail'
  | 'reportDetail'
  | 'trace'
  | 'newReport'

const initialFormState: CaseFormState = {
  failure_type: '',
  failure_description: '',
  part_number: '',
  serial_number: '',
  test_name: '',
  test_value: '',
  spec_lower: '',
  spec_upper: '',
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return null
}

const getReportSummary = (report?: Record<string, unknown>): string => {
  if (!report) {
    return 'No summary available.'
  }

  const explicitSummary = report.summary
  if (typeof explicitSummary === 'string' && explicitSummary.trim()) {
    return explicitSummary
  }

  const topHypothesis = asRecord(report.top_hypothesis)
  if (topHypothesis) {
    const title = topHypothesis.title
    const confidence = topHypothesis.posterior_confidence
    if (typeof title === 'string' && title.trim()) {
      if (typeof confidence === 'number') {
        return `Top hypothesis: ${title} (${Math.round(confidence * 100)}% confidence)`
      }
      return `Top hypothesis: ${title}`
    }
  }

  const sections = Array.isArray(report.sections) ? report.sections : []
  const executiveSummary = sections.find((section) => {
    const sectionRecord = asRecord(section)
    return sectionRecord?.section === 'executive_summary'
  })
  const summaryContent = asRecord(executiveSummary)?.content
  if (typeof summaryContent === 'string' && summaryContent.trim()) {
    return summaryContent.replace(/\s+/g, ' ').slice(0, 260)
  }

  return 'No summary available.'
}

type TraceSegment =
  | { type: 'markdown'; content: string }
  | { type: 'outputData'; language: string; content: string; id: string }

const splitTraceSegments = (trace: string): TraceSegment[] => {
  const segments: TraceSegment[] = []
  const outputDataPattern = /\*\*Output Data:\*\*\s*\n```([a-zA-Z0-9_-]*)\n([\s\S]*?)\n```/g
  let lastIndex = 0
  let blockIndex = 0

  for (const match of trace.matchAll(outputDataPattern)) {
    if (match.index === undefined) {
      continue
    }

    const before = trace.slice(lastIndex, match.index)
    if (before.trim()) {
      segments.push({ type: 'markdown', content: before })
    }

    segments.push({
      type: 'outputData',
      language: match[1] || 'json',
      content: match[2] || '',
      id: `output-data-${blockIndex}`,
    })
    blockIndex += 1
    lastIndex = match.index + match[0].length
  }

  const remaining = trace.slice(lastIndex)
  if (remaining.trim()) {
    segments.push({ type: 'markdown', content: remaining })
  }

  if (segments.length === 0) {
    segments.push({ type: 'markdown', content: trace })
  }

  return segments
}

function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [cases, setCases] = useState<FailureCase[]>([])
  const [reports, setReports] = useState<ReportItem[]>([])
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null)
  const [reportDetail, setReportDetail] = useState<ReportDetail | null>(null)
  const [traceContent, setTraceContent] = useState<string>('')
  const [traceSource, setTraceSource] = useState<string>('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [loading, setLoading] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string>('')
  const [formState, setFormState] = useState<CaseFormState>(initialFormState)
  const [submitMessage, setSubmitMessage] = useState<string>('')
  const [collapsedPanels, setCollapsedPanels] = useState<Record<PanelKey, boolean>>({
    cases: true,
    reports: true,
    caseDetail: true,
    reportDetail: true,
    trace: true,
    newReport: true,
  })

  const togglePanel = (panel: PanelKey) => {
    setCollapsedPanels((prev) => ({ ...prev, [panel]: !prev[panel] }))
  }

  const generateRandomCase = () => {
    const failureTypes = ['leak_test_fail', 'dimensional_oot', 'pressure_drop', 'seal_leak']
    const parts = ['HYD-VALVE-200', 'ACT-PISTON-550', 'PMP-CORE-110', 'REG-BODY-42']
    const testNames = ['Helium_Leak_Test', 'Dim_Check', 'Pressure_Hold', 'Seal_Check']
    const failureType = failureTypes[Math.floor(Math.random() * failureTypes.length)]
    const part = parts[Math.floor(Math.random() * parts.length)]
    const testName = testNames[Math.floor(Math.random() * testNames.length)]
    const testValue = (Math.random() * 5 + 0.1).toFixed(3)
    const specUpper = (Number(testValue) * (Math.random() * 0.4 + 0.6)).toFixed(3)
    const specLower = (Number(specUpper) * 0.6).toFixed(3)

    setFormState({
      failure_type: failureType,
      failure_description: `Automated sample: ${failureType} detected during production validation.`,
      part_number: part,
      serial_number: `SN-${Math.floor(Math.random() * 900000 + 100000)}`,
      test_name: testName,
      test_value: testValue,
      spec_lower: specLower,
      spec_upper: specUpper,
    })
    setSubmitMessage('Random sample loaded. Review and submit when ready.')
  }

  const filteredCases = useMemo(() => {
    if (statusFilter === 'all') {
      return cases
    }
    return cases.filter((item) => item.status === statusFilter)
  }, [cases, statusFilter])

  const selectedCase = useMemo(
    () => cases.find((item) => item.case_id === selectedCaseId) || null,
    [cases, selectedCaseId],
  )

  useEffect(() => {
    refreshAll()
  }, [])

  useEffect(() => {
    if (!selectedReportId) {
      setReportDetail(null)
      setTraceContent('')
      setTraceSource('')
      return
    }
    loadReportDetail(selectedReportId)
    loadReportTrace(selectedReportId)
  }, [selectedReportId])

  const refreshAll = async () => {
    setLoading(true)
    setErrorMessage('')
    try {
      const [metricsResp, casesResp, reportsResp] = await Promise.all([
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/cases`),
        fetch(`${API_BASE}/reports`),
      ])

      if (!metricsResp.ok || !casesResp.ok || !reportsResp.ok) {
        throw new Error('Failed to load dashboard data.')
      }

      const metricsData = await metricsResp.json()
      const casesData = await casesResp.json()
      const reportsData = await reportsResp.json()

      setMetrics(metricsData)
      setCases(casesData.cases || [])
      setReports(reportsData.reports || [])
    } catch (err) {
      setErrorMessage((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const loadReportDetail = async (reportId: string) => {
    try {
      const response = await fetch(`${API_BASE}/reports/${reportId}`)
      if (!response.ok) {
        throw new Error('Failed to load report details.')
      }
      const data = await response.json()
      setReportDetail(data)
    } catch (err) {
      setReportDetail(null)
      setErrorMessage((err as Error).message)
    }
  }

  const loadReportTrace = async (reportId: string) => {
    try {
      const response = await fetch(`${API_BASE}/reports/${reportId}/trace`)
      if (!response.ok) {
        throw new Error('Trace not available for this report.')
      }
      const data = await response.text()
      setTraceContent(data)
      setTraceSource(`Report ${reportId}`)
    } catch (err) {
      setTraceContent('')
      setTraceSource('')
      setErrorMessage((err as Error).message)
    }
  }

  const loadCaseTrace = async (caseId: string) => {
    try {
      const response = await fetch(`${API_BASE}/cases/${caseId}/trace`)
      if (!response.ok) {
        throw new Error('Trace not available for this case.')
      }
      const data = await response.text()
      setTraceContent(data)
      setTraceSource(`Case ${caseId}`)
      setSelectedReportId(null)
      setReportDetail(null)
    } catch (err) {
      setErrorMessage((err as Error).message)
    }
  }

  const handleFormChange = (field: keyof CaseFormState, value: string) => {
    setFormState((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitMessage('')
    setErrorMessage('')

    const payload: Record<string, unknown> = {
      failure_type: formState.failure_type.trim(),
      failure_description: formState.failure_description.trim(),
      part_number: formState.part_number.trim(),
    }

    const optionalFields: Array<keyof CaseFormState> = ['serial_number', 'test_name']
    optionalFields.forEach((field) => {
      const value = formState[field].trim()
      if (value) {
        payload[field] = value
      }
    })

    const numericFields: Array<keyof CaseFormState> = ['test_value', 'spec_lower', 'spec_upper']
    numericFields.forEach((field) => {
      const value = formState[field].trim()
      if (value) {
        payload[field] = Number(value)
      }
    })

    try {
      const response = await fetch(`${API_BASE}/cases`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const errorPayload = await response.json()
        throw new Error(errorPayload.detail || 'Failed to submit case.')
      }
      const data = await response.json()
      setSubmitMessage(`Case submitted: ${data.case_id}`)
      setFormState(initialFormState)
      refreshAll()
    } catch (err) {
      setErrorMessage((err as Error).message)
    }
  }

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div>
          <p className="eyebrow">Manufacturing RCA</p>
          <h1>Failure Report Dashboard</h1>
          <p className="subtitle">
            Monitor case status, review agent reasoning traces, and launch new reports.
          </p>
        </div>
        <div className="header-actions">
          <a className="btn btn--secondary" href="#/flow-lab">
            Open flow lab
          </a>
          <button className="btn btn--secondary" onClick={refreshAll} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh data'}
          </button>
        </div>
      </header>

      {errorMessage && <div className="banner banner--error">{errorMessage}</div>}
      {submitMessage && <div className="banner banner--success">{submitMessage}</div>}

      <section className="metrics-grid">
        <div className="metric-card">
          <p>Total cases</p>
          <h3>{metrics?.cases.total ?? '—'}</h3>
        </div>
        <div className="metric-card">
          <p>Completed</p>
          <h3>{metrics?.cases.completed ?? '—'}</h3>
        </div>
        <div className="metric-card">
          <p>Failed</p>
          <h3>{metrics?.cases.failed ?? '—'}</h3>
        </div>
        <div className="metric-card">
          <p>Pending</p>
          <h3>{metrics?.cases.pending ?? '—'}</h3>
        </div>
        <div className="metric-card">
          <p>Useful feedback</p>
          <h3>{metrics?.feedback.useful_reports ?? '—'}</h3>
        </div>
      </section>

      <main className="dashboard__content">
        <section className="panel">
          <div className="panel__header">
            <h2>Cases</h2>
            <div className="panel__actions">
              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value)}
                className="select"
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="processing">Processing</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
              <button
                type="button"
                className="btn btn--secondary btn--collapse"
                onClick={() => togglePanel('cases')}
              >
                {collapsedPanels.cases ? 'Expand' : 'Collapse'}
              </button>
            </div>
          </div>
          {!collapsedPanels.cases && <div className="panel__body">
            {filteredCases.length === 0 ? (
              <p className="muted">No cases yet.</p>
            ) : (
              <ul className="list">
                {filteredCases.map((item) => (
                  <li
                    key={item.case_id}
                    className={`list-item ${selectedCaseId === item.case_id ? 'is-active' : ''}`}
                    onClick={() => setSelectedCaseId(item.case_id)}
                  >
                    <div>
                      <p className="list-title">{item.case_id}</p>
                      <p className="list-subtitle">{item.submitted_at}</p>
                    </div>
                    <span className={`status-pill status-pill--${item.status}`}>
                      {item.status}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>}
        </section>

        <section className="panel">
          <div className="panel__header">
            <h2>Reports</h2>
            <button
              type="button"
              className="btn btn--secondary btn--collapse"
              onClick={() => togglePanel('reports')}
            >
              {collapsedPanels.reports ? 'Expand' : 'Collapse'}
            </button>
          </div>
          {!collapsedPanels.reports && <div className="panel__body">
            {reports.length === 0 ? (
              <p className="muted">No reports available yet.</p>
            ) : (
              <ul className="list">
                {reports.map((item) => (
                  <li
                    key={item.report_id}
                    className={`list-item ${selectedReportId === item.report_id ? 'is-active' : ''}`}
                    onClick={() => {
                      setSelectedReportId(item.report_id)
                      setSelectedCaseId(item.case_id)
                    }}
                  >
                    <div>
                      <p className="list-title">{item.report_id}</p>
                      <p className="list-subtitle">Case {item.case_id}</p>
                    </div>
                    <span className="status-pill status-pill--completed">completed</span>
                  </li>
                ))}
              </ul>
            )}
          </div>}
        </section>

        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Case detail</h2>
            <div className="panel__actions">
              {selectedCase?.case_id && (
              <button
                className="btn btn--secondary"
                onClick={() => loadCaseTrace(selectedCase.case_id)}
              >
                View case trace
              </button>
              )}
              <button
                type="button"
                className="btn btn--secondary btn--collapse"
                onClick={() => togglePanel('caseDetail')}
              >
                {collapsedPanels.caseDetail ? 'Expand' : 'Collapse'}
              </button>
            </div>
          </div>
          {!collapsedPanels.caseDetail && <div className="panel__body">
            {selectedCase ? (
              <div className="detail-grid">
                <div>
                  <p className="muted">Case ID</p>
                  <p>{selectedCase.case_id}</p>
                </div>
                <div>
                  <p className="muted">Status</p>
                  <p>{selectedCase.status}</p>
                </div>
                <div>
                  <p className="muted">Submitted</p>
                  <p>{selectedCase.submitted_at}</p>
                </div>
                <div>
                  <p className="muted">Report</p>
                  <p>{selectedCase.report_id || '—'}</p>
                </div>
                <div className="detail-wide">
                  <p className="muted">Error</p>
                  <p>{selectedCase.error || '—'}</p>
                </div>
              </div>
            ) : (
              <p className="muted">Select a case to see details.</p>
            )}
          </div>}
        </section>

        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Report detail</h2>
            <button
              type="button"
              className="btn btn--secondary btn--collapse"
              onClick={() => togglePanel('reportDetail')}
            >
              {collapsedPanels.reportDetail ? 'Expand' : 'Collapse'}
            </button>
          </div>
          {!collapsedPanels.reportDetail && <div className="panel__body">
            {reportDetail ? (
              <div className="detail-grid">
                <div>
                  <p className="muted">Report ID</p>
                  <p>{reportDetail.report_id}</p>
                </div>
                <div>
                  <p className="muted">Case ID</p>
                  <p>{reportDetail.case_id}</p>
                </div>
                <div>
                  <p className="muted">Status</p>
                  <p>{reportDetail.status}</p>
                </div>
                <div className="detail-wide">
                  <p className="muted">Summary</p>
                  <p className="muted">
                    {getReportSummary(reportDetail.report)}
                  </p>
                </div>
              </div>
            ) : (
              <p className="muted">Select a report to see details.</p>
            )}
          </div>}
        </section>

        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Reasoning trace</h2>
            <div className="panel__actions">
              <span className="muted">{traceSource || 'No trace loaded.'}</span>
              <button
                type="button"
                className="btn btn--secondary btn--collapse"
                onClick={() => togglePanel('trace')}
              >
                {collapsedPanels.trace ? 'Expand' : 'Collapse'}
              </button>
            </div>
          </div>
          {!collapsedPanels.trace && <div className="panel__body trace-panel">
            {traceContent ? (
              splitTraceSegments(traceContent).map((segment, idx) => {
                if (segment.type === 'outputData') {
                  return (
                    <details key={segment.id} className="trace-output-spoiler">
                      <summary>
                        Output Data ({segment.language}) - click to expand
                      </summary>
                      <pre>
                        <code>{segment.content}</code>
                      </pre>
                    </details>
                  )
                }
                return <ReactMarkdown key={`trace-md-${idx}`}>{segment.content}</ReactMarkdown>
              })
            ) : (
              <p className="muted">Select a report or case trace to view agent reasoning.</p>
            )}
          </div>}
        </section>

        <section className="panel panel--wide">
          <div className="panel__header">
            <h2>Start a new failure report</h2>
            <div className="panel__actions">
              <button type="button" className="btn btn--secondary" onClick={generateRandomCase}>
                Generate random case
              </button>
              <button
                type="button"
                className="btn btn--secondary btn--collapse"
                onClick={() => togglePanel('newReport')}
              >
                {collapsedPanels.newReport ? 'Expand' : 'Collapse'}
              </button>
            </div>
          </div>
          {!collapsedPanels.newReport && <div className="panel__body">
            <form className="form" onSubmit={handleSubmit}>
              <div className="form-grid">
                <label>
                  Failure type
                  <input
                    type="text"
                    required
                    value={formState.failure_type}
                    onChange={(event) => handleFormChange('failure_type', event.target.value)}
                    placeholder="leak_test_fail"
                  />
                </label>
                <label>
                  Part number
                  <input
                    type="text"
                    required
                    value={formState.part_number}
                    onChange={(event) => handleFormChange('part_number', event.target.value)}
                    placeholder="HYD-VALVE-200"
                  />
                </label>
                <label className="form-grid__full">
                  Failure description
                  <textarea
                    required
                    rows={3}
                    value={formState.failure_description}
                    onChange={(event) => handleFormChange('failure_description', event.target.value)}
                    placeholder="Unit failed helium leak test…"
                  />
                </label>
                <label>
                  Serial number
                  <input
                    type="text"
                    value={formState.serial_number}
                    onChange={(event) => handleFormChange('serial_number', event.target.value)}
                  />
                </label>
                <label>
                  Test name
                  <input
                    type="text"
                    value={formState.test_name}
                    onChange={(event) => handleFormChange('test_name', event.target.value)}
                  />
                </label>
                <label>
                  Test value
                  <input
                    type="number"
                    step="any"
                    value={formState.test_value}
                    onChange={(event) => handleFormChange('test_value', event.target.value)}
                  />
                </label>
                <label>
                  Spec lower
                  <input
                    type="number"
                    step="any"
                    value={formState.spec_lower}
                    onChange={(event) => handleFormChange('spec_lower', event.target.value)}
                  />
                </label>
                <label>
                  Spec upper
                  <input
                    type="number"
                    step="any"
                    value={formState.spec_upper}
                    onChange={(event) => handleFormChange('spec_upper', event.target.value)}
                  />
                </label>
              </div>

              <div className="form-actions">
                <button type="submit" className="btn btn--primary">
                  Submit case
                </button>
              </div>
            </form>
          </div>}
        </section>
      </main>
    </div>
  )
}

export default App
