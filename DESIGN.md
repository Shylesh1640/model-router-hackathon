---
version: alpha
name: Model Router — Gothic Brutalism
description: Dark, monolithic visual identity. Gothic weight meets brutalist restraint. No decoration, no curves, maximum contrast.
colors:
  primary: "#0C0C0C"
  secondary: "#4A4A4A"
  tertiary: "#7A1A1A"
  neutral: "#1A1A1A"
  surface: "#121212"
  on-primary: "#E8E6E3"
  on-secondary: "#B0AEAC"
  on-tertiary: "#E8E6E3"
  error: "#9B1B1B"
  code-bg: "#080808"
typography:
  h1:
    fontFamily: Unbounded
    fontSize: 2.5rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.01em"
  h2:
    fontFamily: Unbounded
    fontSize: 1.75rem
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: 0
  h3:
    fontFamily: Unbounded
    fontSize: 1.25rem
    fontWeight: 600
    lineHeight: 1.3
  body-lg:
    fontFamily: Inter
    fontSize: 1.125rem
    fontWeight: 400
    lineHeight: 1.6
  body-md:
    fontFamily: Inter
    fontSize: 1rem
    lineHeight: 1.6
  body-sm:
    fontFamily: Inter
    fontSize: 0.875rem
    lineHeight: 1.5
  label-caps:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: 700
    letterSpacing: "0.12em"
  code:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    lineHeight: 1.5
rounded:
  none: 0px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
  2xl: 80px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    padding: 14px
    typography: "{typography.label-caps}"
  button-primary-hover:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-tertiary}"
  button-secondary:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    padding: 14px
    typography: "{typography.label-caps}"
  card:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    padding: 24px
  card-elevated:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    padding: 32px
  code-block:
    backgroundColor: "{colors.code-bg}"
    textColor: "{colors.on-secondary}"
    rounded: "{rounded.none}"
    typography: "{typography.code}"
    padding: 16px
  input:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    padding: 12px
  input-focus:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
  badge:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    padding: 4px
  badge-fast:
    backgroundColor: "#1A3A1A"
    textColor: "#7ACC7A"
  badge-thinking:
    backgroundColor: "#3A2A1A"
    textColor: "#D4A84B"
  badge-deep:
    backgroundColor: "#2A1A3A"
    textColor: "#B07AD4"
---

## Overview

Gothic brutalism. Heavy blackletter weight meets exposed-concrete severity.
No rounded corners — not one. No shadows — light comes from the screen, not
from fake depth. The palette is three stops on a single axis: black, near-black,
and blood-dried crimson for the single interactive accent.

This is a design system for a model routing dashboard. It should feel like
a control room in a cathedral basement — monolithic, calm, utterly serious.
Every pixel earned its place.

## Colors

- **Primary (#0C0C0C):** Near-black. Page background, input fills, the void.
- **Neutral (#1A1A1A):** Surface colour for cards and panels. One step off
  primary to create hierarchy without decoration.
- **Surface (#121212):** The exact dark used for the main content area. Sits
  between primary and neutral.
- **Secondary (#4A4A4A):** Muted text, captions, placeholder content. Concrete
  grey, never warm.
- **Tertiary (#7A1A1A):** The single accent — dried crimson. Buttons,
  badges, focus states, the only colour on the page that moves.
- **Border (#2A2A2A):** Structural dividers. Visible but not loud.
- **Error (#9B1B1B):** Slightly hotter red than tertiary. Reserved for
  destructive actions and errors.
- **On- colours:** `#E8E6E3` is a warm off-white — easier on the eyes at
  high contrast than pure white `#FFFFFF`.

## Typography

Two-font system with a dedicated code face.

**Unbounded** (headings) — Heavy geometric sans with blackletter influence in
the terminals. Use for all display sizes. Tighter letter-spacing on h1 to
compress the mass. No letter-spacing adjustment on h2.

**Inter** (body) — Utilitarian, readable, maximally neutral. Body-lg for
lead paragraphs on the dashboard. Body-sm for metadata and timestamps.
Label-caps is the only uppercase treatment — 0.12em tracking, bold weight.
Use exclusively for button labels and section headers.

**JetBrains Mono** (code) — Developer default. Used in the live feed entries,
model IDs, and any token/code display. The only monospace in the system.

No font-weight below 400 anywhere. Gothic brutalism is heavy or it's nothing.

## Layout

The layout grid uses a 4px baseline. The spacing scale is deliberately
limited — brutalist architecture repeats the same module everywhere.

- `md` (16px) is the atomic unit: intra-card gaps, icon spacing, input padding.
- `lg` (24px) for inter-component gaps between cards and panels.
- `xl` (48px) for section breaks and modal padding.
- `2xl` (80px) for page-level margins.

Full-width panels are preferred over sidebars. Information density is high
but regimented — the live feed uses uniform row heights.

## Elevation & Depth

No elevation. No shadows. Depth is expressed through value contrast —
`{colors.primary}` (page) → `{colors.surface}` (content area) →
`{colors.neutral}` (card). That's the entire depth stack.

The only "lifted" state is `card-elevated`, which increases padding to 32px
and may use a slightly thinner border. It does not use a drop shadow.

## Shapes

Zero. Square everything. `rounded: none (0px)` is the only radius in the
system. Brutalism doesn't curve.

## Components

- **`button-primary`** — The single interactive call to action per view.
  All-caps label. Dried crimson fill. No border. Hover shifts to black fill.
- **`button-secondary`** — Transparent fill, secondary text colour. For
  dismissible or non-primary actions. Same all-caps label.
- **`card`** — Default surface for grouped content. 24px padding, dark
  neutral fill, pinned to the border colour. No shadow, no rounding.
- **`card-elevated`** — 32px padding for hero or focused content. Same
  visual weight, more breathing room.
- **`code-block`** — Near-black background (`#080808`), secondary text.
  JetBrains Mono. Used for routing logs, model IDs, response previews.
- **`input`** — Same background as the page (`{colors.primary}`). Only
  distinguished by the border colour. Focus shifts border to tertiary.
- **`badge`** — Status indicators. Three variants:
  - `badge` (default): tertiary fill, white text — untyped status.
  - `badge-fast`: dark green fill, light green text.
  - `badge-thinking`: dark amber fill, light amber text.
  - `badge-deep`: dark purple fill, light purple text.

## Do's and Don'ts

- **Do** use the crimson (`{colors.tertiary}`) sparingly. One element per
  view should carry it. If everything is accent, nothing is.
- **Don't** add rounded corners. `rounded: none` is a binding constraint.
- **Don't** introduce shadows or gradients. Brutalism is flat. Light comes
  from the screen.
- **Do** use `{colors.code-bg}` (`#080808`) for code and data displays.
  It's the only true black in the system — reserved for raw data surfaces.
- **Don't** use white (`#FFFFFF`) as a text colour. `{colors.on-primary}`
  (`#E8E6E3`) is warmer and more legible at high contrast.
- **Don't** add a third interactive colour. Tertiary is the only accent.
  Secondary actions use `button-secondary` with transparent fill.
- **Do** use `label-caps` for all button labels — lowercase or mixed-case
  on buttons violates the brutalist mandate.
- **Don't** use font-weight below 400. Gothic weight requires gravity.
