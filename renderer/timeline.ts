import { CarbonPathway, PathNode } from "../schema/pathway.schema";

export type TimelineRenderOptions = {
  width?: number;
  height?: number;
  padding?: number;
  fontFamily?: string;
};

const DEFAULT_OPTIONS: Required<TimelineRenderOptions> = {
  width: 1200,
  height: 600,
  padding: 80,
  fontFamily: "'Noto Sans', 'PingFang SC', sans-serif",
};

const CARD_WIDTH = 220;
const CARD_HEIGHT = 90;
const AXIS_Y = 300;

const yearToX = (
  year: number,
  pathway: CarbonPathway,
  options: Required<TimelineRenderOptions>,
): number => {
  const usableWidth = options.width - options.padding * 2;
  return (
    options.padding +
    ((year - pathway.baseline_year) / (pathway.neutral_year - pathway.baseline_year)) *
      usableWidth
  );
};

const renderCardText = (node: PathNode): string => {
  const actionLines = node.actions
    .slice(0, 3)
    .map(
      (action, index) =>
        `<tspan x="0" dy="${index === 0 ? 18 : 16}">• ${action}</tspan>`,
    )
    .join("");

  return `<text font-size="12" fill="#1F2937">
    <tspan x="0" dy="0" font-size="14" font-weight="600">${node.title}</tspan>
    ${actionLines}
  </text>`;
};

const renderNode = (
  node: PathNode,
  index: number,
  pathway: CarbonPathway,
  options: Required<TimelineRenderOptions>,
): string => {
  const x = yearToX(node.year, pathway, options);
  const cardAbove = index % 2 === 0;
  const cardX = x - CARD_WIDTH / 2;
  const cardY = cardAbove ? AXIS_Y - 160 : AXIS_Y + 70;
  const lineY = cardAbove ? AXIS_Y - 20 : AXIS_Y + 20;
  const lineEndY = cardAbove ? cardY + CARD_HEIGHT : cardY;

  return `
    <g class="node" data-stage="${node.stage}">
      <line x1="${x}" y1="${lineY}" x2="${x}" y2="${lineEndY}" stroke="#94A3B8" stroke-width="2" />
      <circle cx="${x}" cy="${AXIS_Y}" r="8" fill="#10B981" />
      <rect x="${cardX}" y="${cardY}" width="${CARD_WIDTH}" height="${CARD_HEIGHT}" rx="12" fill="#FFFFFF" stroke="#E2E8F0" />
      <g transform="translate(${cardX + 16}, ${cardY + 20})">
        ${renderCardText(node)}
      </g>
      <text x="${x}" y="${cardAbove ? cardY - 12 : cardY + CARD_HEIGHT + 18}" text-anchor="middle" font-size="12" fill="#64748B">
        ${node.year}
      </text>
    </g>
  `;
};

const renderMascot = (
  node: PathNode,
  pathway: CarbonPathway,
  options: Required<TimelineRenderOptions>,
): string => {
  const x = yearToX(node.year, pathway, options) - 24;
  const y = AXIS_Y - 120;
  return `<image href="mascot.svg" x="${x}" y="${y}" width="48" height="48" />`;
};

export const renderTimelineSvg = (
  pathway: CarbonPathway,
  options: TimelineRenderOptions = {},
): string => {
  const merged = { ...DEFAULT_OPTIONS, ...options };

  const nodesMarkup = pathway.nodes
    .map((node, index) => renderNode(node, index, pathway, merged))
    .join("");

  const mascotNodes = pathway.nodes.filter(
    (node) => node.stage === "baseline" || node.stage === "peak" || node.stage === "neutral",
  );

  const mascotMarkup = mascotNodes
    .map((node) => renderMascot(node, pathway, merged))
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${merged.width}" height="${merged.height}" viewBox="0 0 ${merged.width} ${merged.height}" font-family=${merged.fontFamily}>
  <g id="background">
    <rect width="100%" height="100%" fill="#F8FAFC" />
  </g>
  <g id="pathway">
    <line x1="${merged.padding}" y1="${AXIS_Y}" x2="${merged.width - merged.padding}" y2="${AXIS_Y}" stroke="#0F172A" stroke-width="3" />
  </g>
  <g id="nodes">
    ${nodesMarkup}
  </g>
  <g id="mascot">
    ${mascotMarkup}
  </g>
  <g id="labels">
    <text x="${merged.padding}" y="40" font-size="22" font-weight="700" fill="#0F172A">ESG Carbon Pathway</text>
  </g>
</svg>`;
};
