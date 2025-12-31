import { generateCarbonPathway } from "../agent/pathwayAgent";
import { DocumentInput, UserConfig, validateCarbonPathway } from "../schema/pathway.schema";
import { renderTimelineSvg } from "../renderer/timeline";
import { renderStepSvg } from "../renderer/step";
import { renderMountainSvg } from "../renderer/mountain";

export type GeneratePathwayRequest = {
  document: DocumentInput;
  config?: UserConfig;
};

export type GeneratePathwayResponse = {
  pathway: ReturnType<typeof generateCarbonPathway>;
  svg: string;
  validation_errors: string[];
};

const renderSvgByStyle = (pathwayStyle: UserConfig["pathway_style"], pathway: ReturnType<typeof generateCarbonPathway>): string => {
  switch (pathwayStyle) {
    case "step":
      return renderStepSvg(pathway);
    case "mountain":
      return renderMountainSvg(pathway);
    case "timeline":
    default:
      return renderTimelineSvg(pathway);
  }
};

export const generatePathway = (
  request: GeneratePathwayRequest,
): GeneratePathwayResponse => {
  const pathway = generateCarbonPathway(request.document, request.config);
  const validationErrors = validateCarbonPathway(pathway);
  const svg = renderSvgByStyle(request.config?.pathway_style ?? "timeline", pathway);

  return {
    pathway,
    svg,
    validation_errors: validationErrors,
  };
};
