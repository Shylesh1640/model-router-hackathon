import { useMemo, useState } from "react";
import {
  Activity,
  Bell,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  CreditCard,
  ExternalLink,
  Eye,
  EyeOff,
  Globe,
  KeyRound,
  LayoutDashboard,
  Moon,
  Search,
  Settings,
  Sun,
  WalletCards,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Page = "Dashboard" | "Models" | "Billing" | "Settings";
type ThemeMode = "dark" | "light";

const MODEL_ROWS = [
  { provider: "OpenAI", model: "gpt-4.1", usage: 18.4, input: 2.0, output: 8.0, requests: 46200 },
  { provider: "OpenAI", model: "gpt-4.1-mini", usage: 64.8, input: 0.4, output: 1.6, requests: 184900 },
  { provider: "OpenAI", model: "gpt-4o", usage: 12.7, input: 2.5, output: 10.0, requests: 31800 },
  { provider: "Anthropic", model: "claude-3.5-sonnet", usage: 21.6, input: 3.0, output: 15.0, requests: 54100 },
  { provider: "Anthropic", model: "claude-3-haiku", usage: 96.3, input: 0.25, output: 1.25, requests: 219400 },
  { provider: "Google", model: "gemini-1.5-pro", usage: 10.2, input: 1.25, output: 5.0, requests: 24650 },
  { provider: "Meta", model: "llama-3.1-70b-instruct", usage: 33.9, input: 0.72, output: 0.72, requests: 88700 },
  { provider: "Mistral", model: "mistral-large", usage: 8.8, input: 2.0, output: 6.0, requests: 19300 },
];

const CREDIT_PACKS = [
  { label: "Opening credit", credits: 500000, cost: 500 },
  { label: "July top-up", credits: 250000, cost: 250 },
  { label: "Enterprise promo", credits: 100000, cost: 0 },
];

const TIER_DATA = [
  { name: "Easy", value: 96.3, color: "#7c3aed" },
  { name: "Medium", value: 42.0, color: "#0ea5e9" },
  { name: "Difficult", value: 31.1, color: "#f97316" },
];

const NAV_ITEMS: { icon: typeof LayoutDashboard; label: Page }[] = [
  { icon: LayoutDashboard, label: "Dashboard" },
  { icon: Cpu, label: "Models" },
  { icon: CreditCard, label: "Billing" },
  { icon: Settings, label: "Settings" },
];

function currency(value: number) {
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function millionTokens(value: number) {
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}M`;
}

function App() {
  const [page, setPage] = useState<Page>("Dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [showKey, setShowKey] = useState(false);
  const [website, setWebsite] = useState("https://openrouter.ai");
  const [apiKey, setApiKey] = useState("sk-or-v1-prod-routeiq-placeholder");

  const dark = theme === "dark";
  const palette = {
    bg: dark ? "#08090b" : "#f7f8fb",
    panel: dark ? "rgba(16,18,22,0.88)" : "rgba(255,255,255,0.90)",
    panelSoft: dark ? "rgba(23,26,32,0.86)" : "rgba(238,242,247,0.88)",
    text: dark ? "#f8fafc" : "#111827",
    muted: dark ? "#98a2b3" : "#667085",
    line: dark ? "rgba(255,255,255,0.10)" : "rgba(17,24,39,0.12)",
    accent: dark ? "#f8fafc" : "#111827",
    brand: "#7c3aed",
    good: "#16a34a",
  };
  const sidebarW = collapsed ? 64 : 220;

  const totals = useMemo(() => {
    const usage = MODEL_ROWS.reduce((sum, row) => sum + row.usage, 0);
    const modelCost = MODEL_ROWS.reduce((sum, row) => {
      const inputShare = row.usage * 0.62;
      const outputShare = row.usage * 0.38;
      return sum + inputShare * row.input + outputShare * row.output;
    }, 0);
    const creditCost = CREDIT_PACKS.reduce((sum, row) => sum + row.cost, 0);
    const credits = CREDIT_PACKS.reduce((sum, row) => sum + row.credits, 0);
    return { usage, modelCost, creditCost, credits };
  }, []);

  const pageTitle = page === "Dashboard" ? "Intelligent Routing Analytics" : page;

  return (
    <div style={{ minHeight: "100vh", background: palette.bg, color: palette.text, fontFamily: "Inter, ui-sans-serif, system-ui", position: "relative", overflowX: "hidden" }}>
      <style>{`
        * { box-sizing: border-box; }
        button, input { font: inherit; }
        table { border-collapse: collapse; width: 100%; }
        @media (max-width: 920px) {
          .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr !important; }
          .desktop-search { display: none !important; }
        }
      `}</style>

      <aside style={{ position: "fixed", inset: "0 auto 0 0", width: sidebarW, borderRight: `1px solid ${palette.line}`, background: palette.panel, backdropFilter: "blur(18px)", transition: "width 160ms ease", zIndex: 10 }}>
        <div style={{ height: 64, display: "flex", alignItems: "center", gap: 12, padding: 16, borderBottom: `1px solid ${palette.line}` }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, display: "grid", placeItems: "center", background: palette.brand, color: "white", fontWeight: 800 }}>IR</div>
          {!collapsed && <strong style={{ letterSpacing: 0 }}>RouteIQ</strong>}
        </div>

        <nav style={{ padding: 10, display: "grid", gap: 6 }}>
          {NAV_ITEMS.map(({ icon: Icon, label }) => {
            const active = page === label;
            return (
              <button
                key={label}
                title={collapsed ? label : undefined}
                onClick={() => setPage(label)}
                style={{
                  border: `1px solid ${active ? palette.line : "transparent"}`,
                  background: active ? palette.panelSoft : "transparent",
                  color: active ? palette.text : palette.muted,
                  borderRadius: 8,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  minHeight: 42,
                  padding: "0 12px",
                  width: "100%",
                }}
              >
                <Icon size={18} />
                {!collapsed && <span>{label}</span>}
              </button>
            );
          })}
        </nav>

        <button
          onClick={() => setCollapsed((value) => !value)}
          style={{ position: "absolute", left: 12, bottom: 12, width: "calc(100% - 24px)", height: 38, borderRadius: 8, border: `1px solid ${palette.line}`, background: "transparent", color: palette.muted, cursor: "pointer" }}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </aside>

      <header style={{ position: "fixed", left: sidebarW, right: 0, top: 0, height: 64, display: "flex", alignItems: "center", gap: 14, padding: "0 24px", borderBottom: `1px solid ${palette.line}`, background: palette.panel, backdropFilter: "blur(18px)", transition: "left 160ms ease", zIndex: 9 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, lineHeight: 1.2 }}>{pageTitle}</h1>
          <div style={{ color: palette.muted, fontSize: 12 }}>Jul 4, 2026 · OpenRouter workspace</div>
        </div>
        <div style={{ flex: 1 }} />
        <div className="desktop-search" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 8, background: palette.panelSoft, color: palette.muted }}>
          <Search size={15} />
          <span style={{ fontSize: 13 }}>Search models, usage, invoices</span>
        </div>
        <Bell size={18} color={palette.muted} />
      </header>

      <main style={{ marginLeft: sidebarW, paddingTop: 64, transition: "margin-left 160ms ease", position: "relative", zIndex: 1 }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: 24 }}>
          {page === "Dashboard" && <Dashboard palette={palette} totals={totals} />}
          {page === "Models" && <Models palette={palette} />}
          {page === "Billing" && <Billing palette={palette} totals={totals} />}
          {page === "Settings" && (
            <SettingsPage
              palette={palette}
              theme={theme}
              setTheme={setTheme}
              website={website}
              setWebsite={setWebsite}
              apiKey={apiKey}
              setApiKey={setApiKey}
              showKey={showKey}
              setShowKey={setShowKey}
            />
          )}
        </div>
      </main>
    </div>
  );
}

function Card({ palette, children, style }: { palette: any; children: React.ReactNode; style?: React.CSSProperties }) {
  return <section style={{ background: palette.panel, border: `1px solid ${palette.line}`, borderRadius: 8, padding: 20, ...style }}>{children}</section>;
}

function Metric({ palette, label, value, sub, icon: Icon }: { palette: any; label: string; value: string; sub: string; icon: typeof Activity }) {
  return (
    <Card palette={palette}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ color: palette.muted, fontSize: 12 }}>{label}</div>
          <div style={{ fontSize: 28, fontWeight: 800, marginTop: 8 }}>{value}</div>
        </div>
        <Icon size={22} color={palette.brand} />
      </div>
      <div style={{ color: palette.muted, fontSize: 12, marginTop: 12 }}>{sub}</div>
    </Card>
  );
}

function Dashboard({ palette, totals }: { palette: any; totals: { usage: number; modelCost: number; creditCost: number; credits: number } }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Card palette={palette}>
        <div style={{ display: "grid", gap: 16 }}>
          <div>
            <div style={{ color: palette.muted, fontSize: 12, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase" }}>Routing overview</div>
            <div style={{ fontSize: 34, lineHeight: 1.05, fontWeight: 850, maxWidth: 520, marginTop: 10 }}>Static routing signal monitor</div>
            <p style={{ margin: "12px 0 0", color: palette.muted, fontSize: 14, lineHeight: 1.7, maxWidth: 620 }}>
              Model routing, credit status, and OpenRouter connection health are shown without any moving background effects.
            </p>
          </div>
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
            {[
              ["Classifier", "Online"],
              ["Confidence", "96.3%"],
              ["OpenRouter", "Linked"],
            ].map(([label, value]) => (
              <div key={label} style={{ border: `1px solid ${palette.line}`, borderRadius: 8, padding: 16, background: palette.panelSoft }}>
                <div style={{ color: palette.muted, fontSize: 12 }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 800, marginTop: 6 }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <div className="grid-4" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <Metric palette={palette} label="Total model usage" value={millionTokens(totals.usage)} sub="Measured in million tokens" icon={Cpu} />
        <Metric palette={palette} label="Model run cost" value={currency(totals.modelCost)} sub="Blended input and output spend" icon={Activity} />
        <Metric palette={palette} label="Credit balance" value="587K" sub={`${totals.credits.toLocaleString()} total credits purchased`} icon={WalletCards} />
        <Metric palette={palette} label="Routing pages" value="4" sub="Routing logs page removed" icon={CheckCircle2} />
      </div>

      <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card palette={palette}>
          <h2 style={{ margin: "0 0 16px", fontSize: 16 }}>Usage by difficulty</h2>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={TIER_DATA} dataKey="value" nameKey="name" innerRadius={68} outerRadius={100} paddingAngle={4}>
                {TIER_DATA.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
              </Pie>
              <Tooltip formatter={(value: number) => `${value}M tokens`} />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card palette={palette}>
          <h2 style={{ margin: "0 0 16px", fontSize: 16 }}>Top model usage</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={MODEL_ROWS.slice(0, 6)}>
              <CartesianGrid stroke={palette.line} vertical={false} />
              <XAxis dataKey="model" tick={{ fill: palette.muted, fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: palette.muted, fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip formatter={(value: number) => `${value}M tokens`} />
              <Bar dataKey="usage" fill={palette.brand} radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

function Models({ palette }: { palette: any }) {
  return (
    <Card palette={palette}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, marginBottom: 18 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 18 }}>All models</h2>
          <p style={{ margin: "6px 0 0", color: palette.muted, fontSize: 13 }}>Usage is shown in million tokens. Costs are per million tokens.</p>
        </div>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr style={{ color: palette.muted, fontSize: 12, textAlign: "left", borderBottom: `1px solid ${palette.line}` }}>
              <th style={{ padding: "12px 10px" }}>Provider</th>
              <th style={{ padding: "12px 10px" }}>Model</th>
              <th style={{ padding: "12px 10px" }}>Usage</th>
              <th style={{ padding: "12px 10px" }}>Input cost / 1M</th>
              <th style={{ padding: "12px 10px" }}>Output cost / 1M</th>
              <th style={{ padding: "12px 10px" }}>Requests</th>
              <th style={{ padding: "12px 10px" }}>Estimated cost</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_ROWS.map((row) => {
              const estimated = row.usage * 0.62 * row.input + row.usage * 0.38 * row.output;
              return (
                <tr key={row.model} style={{ borderBottom: `1px solid ${palette.line}` }}>
                  <td style={{ padding: "14px 10px", color: palette.muted }}>{row.provider}</td>
                  <td style={{ padding: "14px 10px", fontWeight: 700 }}>{row.model}</td>
                  <td style={{ padding: "14px 10px" }}>{millionTokens(row.usage)}</td>
                  <td style={{ padding: "14px 10px" }}>{currency(row.input)}</td>
                  <td style={{ padding: "14px 10px" }}>{currency(row.output)}</td>
                  <td style={{ padding: "14px 10px" }}>{row.requests.toLocaleString()}</td>
                  <td style={{ padding: "14px 10px", fontWeight: 700 }}>{currency(estimated)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Billing({ palette, totals }: { palette: any; totals: { usage: number; modelCost: number; creditCost: number; credits: number } }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        <Metric palette={palette} label="Total credit cost" value={currency(totals.creditCost)} sub={`${totals.credits.toLocaleString()} credits purchased`} icon={CreditCard} />
        <Metric palette={palette} label="Model usage cost" value={currency(totals.modelCost)} sub={`${millionTokens(totals.usage)} consumed this period`} icon={Cpu} />
        <Metric palette={palette} label="Remaining credit value" value={currency(587)} sub="Estimated from active credit balance" icon={WalletCards} />
      </div>

      <Card palette={palette}>
        <h2 style={{ margin: "0 0 16px", fontSize: 18 }}>Credit purchases</h2>
        <table>
          <thead>
            <tr style={{ color: palette.muted, fontSize: 12, textAlign: "left", borderBottom: `1px solid ${palette.line}` }}>
              <th style={{ padding: "12px 10px" }}>Credit item</th>
              <th style={{ padding: "12px 10px" }}>Credits</th>
              <th style={{ padding: "12px 10px" }}>Cost</th>
            </tr>
          </thead>
          <tbody>
            {CREDIT_PACKS.map((row) => (
              <tr key={row.label} style={{ borderBottom: `1px solid ${palette.line}` }}>
                <td style={{ padding: "14px 10px", fontWeight: 700 }}>{row.label}</td>
                <td style={{ padding: "14px 10px" }}>{row.credits.toLocaleString()}</td>
                <td style={{ padding: "14px 10px" }}>{currency(row.cost)}</td>
              </tr>
            ))}
            <tr>
              <td style={{ padding: "14px 10px", fontWeight: 800 }}>Total</td>
              <td style={{ padding: "14px 10px", fontWeight: 800 }}>{totals.credits.toLocaleString()}</td>
              <td style={{ padding: "14px 10px", fontWeight: 800 }}>{currency(totals.creditCost)}</td>
            </tr>
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function SettingsPage({
  palette,
  theme,
  setTheme,
  website,
  setWebsite,
  apiKey,
  setApiKey,
  showKey,
  setShowKey,
}: {
  palette: any;
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  website: string;
  setWebsite: (value: string) => void;
  apiKey: string;
  setApiKey: (value: string) => void;
  showKey: boolean;
  setShowKey: (value: boolean) => void;
}) {
  return (
    <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <Card palette={palette}>
        <h2 style={{ margin: "0 0 18px", fontSize: 18 }}>Appearance</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {[
            { label: "Black theme", value: "dark" as ThemeMode, icon: Moon },
            { label: "White theme", value: "light" as ThemeMode, icon: Sun },
          ].map(({ label, value, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setTheme(value)}
              style={{ padding: 16, borderRadius: 8, border: `1px solid ${theme === value ? palette.brand : palette.line}`, background: theme === value ? palette.panelSoft : "transparent", color: palette.text, cursor: "pointer", display: "flex", alignItems: "center", gap: 10 }}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </div>
      </Card>

      <Card palette={palette}>
        <h2 style={{ margin: "0 0 18px", fontSize: 18 }}>OpenRouter connection</h2>
        <label style={{ color: palette.muted, fontSize: 13 }}>Website link</label>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, marginBottom: 18 }}>
          <Globe size={17} color={palette.muted} />
          <input value={website} onChange={(event) => setWebsite(event.target.value)} style={{ flex: 1, border: `1px solid ${palette.line}`, borderRadius: 8, padding: "10px 12px", background: palette.panelSoft, color: palette.text, minWidth: 0 }} />
          <a href={website} target="_blank" rel="noreferrer" style={{ color: palette.brand, display: "inline-flex" }}><ExternalLink size={18} /></a>
        </div>

        <label style={{ color: palette.muted, fontSize: 13 }}>OpenRouter API key</label>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
          <KeyRound size={17} color={palette.muted} />
          <input
            type={showKey ? "text" : "password"}
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            style={{ flex: 1, border: `1px solid ${palette.line}`, borderRadius: 8, padding: "10px 12px", background: palette.panelSoft, color: palette.text, minWidth: 0 }}
          />
          <button onClick={() => setShowKey(!showKey)} style={{ border: `1px solid ${palette.line}`, borderRadius: 8, width: 42, height: 42, background: "transparent", color: palette.muted, cursor: "pointer" }}>
            {showKey ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        </div>
      </Card>
    </div>
  );
}

export default App;
