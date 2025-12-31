import { CarbonPathway } from "../schema/pathway.schema";

const buildPoints = (pathway: CarbonPathway, width: number, height: number): string => {
  const padding = 80;
  const usableWidth = width - padding * 2;
  const baseY = height - 120;
  const maxRise = 220;

  return pathway.nodes
    .map((node, index) => {
      const x = padding + (index / (pathway.nodes.length - 1)) * usableWidth;
      const isPeak = node.stage === "peak";
      const isEdge = node.stage === "baseline" || node.stage === "neutral";
      const y = isPeak ? baseY - maxRise : isEdge ? baseY : baseY - maxRise * 0.6;
      return `${x},${y}`;
    })
    .join(" ");
};

export const renderMountainSvg = (pathway: CarbonPathway): string => {
  const width = 1200;
  const height = 600;
  const points = buildPoints(pathway, width, height);

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <g id="background">
    <rect width="100%" height="100%" fill="#F8FAFC" />
  </g>
  <g id="pathway">
    <polyline points="${points}" fill="none" stroke="#0EA5E9" stroke-width="4" />
  </g>
</svg>`;
};
