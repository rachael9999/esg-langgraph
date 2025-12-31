import { CarbonPathway } from "../schema/pathway.schema";

export const renderStepSvg = (pathway: CarbonPathway): string => {
  const width = 1200;
  const height = 600;
  const stepHeight = 60;
  const stepWidth = (width - 160) / pathway.nodes.length;

  const steps = pathway.nodes
    .map((node, index) => {
      const x = 80 + index * stepWidth;
      const y = 420 - index * stepHeight;
      return `
        <rect x="${x}" y="${y}" width="${stepWidth - 16}" height="${stepHeight}" fill="#E2E8F0" />
        <text x="${x + 8}" y="${y + 24}" font-size="12" fill="#0F172A">${node.year}</text>
        <text x="${x + 8}" y="${y + 44}" font-size="12" fill="#475569">${node.title}</text>
      `;
    })
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <g id="background">
    <rect width="100%" height="100%" fill="#F8FAFC" />
  </g>
  <g id="pathway">
    ${steps}
  </g>
</svg>`;
};
