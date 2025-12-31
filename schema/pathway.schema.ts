export type DocumentInput = {
  raw_text: string;
  language?: "zh" | "en";
};

export type UserConfig = {
  peak_year?: number;
  neutral_year?: number;
  pathway_style?: "timeline" | "step" | "mountain";
  tone?: "formal" | "public";
};

export type PathNodeStage =
  | "baseline"
  | "short_term"
  | "peak"
  | "mid_term"
  | "long_term"
  | "neutral";

export type PathNodeCategory =
  | "data"
  | "energy"
  | "efficiency"
  | "process"
  | "supply_chain"
  | "offset";

export type PathNode = {
  id: string;
  year: number;
  stage: PathNodeStage;
  title: string;
  actions: string[];
  category: PathNodeCategory;
  explanation: string;
  source: "document" | "inferred";
};

export type CarbonPathway = {
  baseline_year: number;
  peak_year: number;
  neutral_year: number;
  assumptions: string[];
  nodes: PathNode[];
};

const MAX_TITLE_LENGTH = 12;
const MAX_ACTION_LENGTH = 16;
const MAX_ACTIONS = 3;

export const validateCarbonPathway = (pathway: CarbonPathway): string[] => {
  const errors: string[] = [];

  if (!pathway.baseline_year) {
    errors.push("baseline_year is required");
  }
  if (!pathway.peak_year) {
    errors.push("peak_year is required");
  }
  if (!pathway.neutral_year) {
    errors.push("neutral_year is required");
  }
  if (pathway.peak_year >= pathway.neutral_year) {
    errors.push("peak_year must be less than neutral_year");
  }

  const years = pathway.nodes.map((node) => node.year);
  for (let i = 1; i < years.length; i += 1) {
    if (years[i] <= years[i - 1]) {
      errors.push("node years must be strictly increasing");
      break;
    }
  }

  const peakNodes = pathway.nodes.filter((node) => node.stage === "peak");
  const neutralNodes = pathway.nodes.filter((node) => node.stage === "neutral");
  if (peakNodes.length !== 1) {
    errors.push("pathway must contain exactly one peak node");
  }
  if (neutralNodes.length !== 1) {
    errors.push("pathway must contain exactly one neutral node");
  }

  pathway.nodes.forEach((node) => {
    if (node.title.length > MAX_TITLE_LENGTH) {
      errors.push(`title too long for node ${node.id}`);
    }
    if (node.actions.length > MAX_ACTIONS) {
      errors.push(`too many actions for node ${node.id}`);
    }
    node.actions.forEach((action) => {
      if (action.length > MAX_ACTION_LENGTH) {
        errors.push(`action too long for node ${node.id}`);
      }
    });
  });

  return errors;
};
