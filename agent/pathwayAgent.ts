import {
  CarbonPathway,
  DocumentInput,
  PathNode,
  PathNodeCategory,
  PathNodeStage,
  UserConfig,
} from "../schema/pathway.schema";

const DEFAULT_BASELINE_YEAR = new Date().getFullYear();
const DEFAULT_PEAK_YEAR = 2030;
const DEFAULT_NEUTRAL_YEAR = 2050;

const STAGE_ORDER: PathNodeStage[] = [
  "baseline",
  "short_term",
  "peak",
  "mid_term",
  "long_term",
  "neutral",
];

const CATEGORY_ROTATION: PathNodeCategory[] = [
  "data",
  "energy",
  "efficiency",
  "process",
  "supply_chain",
  "offset",
];

const inferYearsFromText = (rawText: string): number[] => {
  const matches = rawText.match(/(20\d{2})/g) ?? [];
  return matches
    .map((value) => Number(value))
    .filter((year) => year >= 2000 && year <= 2100)
    .sort((a, b) => a - b);
};

const selectPeakYear = (years: number[], config?: UserConfig): number => {
  if (config?.peak_year) {
    return config.peak_year;
  }
  const candidate = years.find((year) => year >= DEFAULT_BASELINE_YEAR && year <= 2035);
  return candidate ?? DEFAULT_PEAK_YEAR;
};

const selectNeutralYear = (years: number[], config?: UserConfig): number => {
  if (config?.neutral_year) {
    return config.neutral_year;
  }
  const candidate = [...years].reverse().find((year) => year >= 2040);
  return candidate ?? DEFAULT_NEUTRAL_YEAR;
};

const buildNodes = (
  baselineYear: number,
  peakYear: number,
  neutralYear: number,
): PathNode[] => {
  const stages = STAGE_ORDER;
  const years = [
    baselineYear,
    Math.round((baselineYear + peakYear) / 2),
    peakYear,
    Math.round((peakYear + neutralYear) / 2),
    neutralYear - 5,
    neutralYear,
  ].map((year, index, array) => {
    if (index === 0) {
      return year;
    }
    const prev = array[index - 1];
    return year <= prev ? prev + 1 : year;
  });

  return stages.map((stage, index) => ({
    id: `n${index + 1}`,
    year: years[index],
    stage,
    title: titleForStage(stage),
    actions: actionsForStage(stage),
    category: CATEGORY_ROTATION[index % CATEGORY_ROTATION.length],
    explanation: explanationForStage(stage),
    source: "inferred",
  }));
};

const titleForStage = (stage: PathNodeStage): string => {
  switch (stage) {
    case "baseline":
      return "建立碳盘查";
    case "short_term":
      return "快速节能";
    case "peak":
      return "实现达峰";
    case "mid_term":
      return "结构优化";
    case "long_term":
      return "深度减排";
    case "neutral":
      return "实现中和";
    default:
      return "路径节点";
  }
};

const actionsForStage = (stage: PathNodeStage): string[] => {
  switch (stage) {
    case "baseline":
      return ["统一口径", "完善盘查"]; 
    case "short_term":
      return ["能效改造", "绿色采购"]; 
    case "peak":
      return ["减排兑现", "稳控增长"]; 
    case "mid_term":
      return ["工艺升级", "替代能源"]; 
    case "long_term":
      return ["供应链协同", "深化减排"]; 
    case "neutral":
      return ["残余抵消", "持续运营"]; 
    default:
      return ["行动计划"]; 
  }
};

const explanationForStage = (stage: PathNodeStage): string => {
  switch (stage) {
    case "baseline":
      return "建立可靠的碳数据基础，为后续减排制定基准。";
    case "short_term":
      return "先做容易落地的节能与管理优化，形成早期成果。";
    case "peak":
      return "通过结构调整与控制增量排放，实现排放峰值。";
    case "mid_term":
      return "推动核心工艺和能源结构升级，持续下降排放。";
    case "long_term":
      return "深化供应链协同与技术替代，锁定长期减排。";
    case "neutral":
      return "通过残余排放抵消与持续优化，实现碳中和目标。";
    default:
      return "说明占位。";
  }
};

export const generateCarbonPathway = (
  document: DocumentInput,
  config?: UserConfig,
): CarbonPathway => {
  const yearsFromText = inferYearsFromText(document.raw_text);
  const baselineYear = yearsFromText[0] ?? DEFAULT_BASELINE_YEAR;
  const peakYear = selectPeakYear(yearsFromText, config);
  const neutralYear = selectNeutralYear(yearsFromText, config);

  const assumptions = [
    document.language === "en"
      ? "No sector specified, using a generic corporate decarbonization pathway."
      : "未明确行业，采用通用企业双碳路径假设",
  ];

  const nodes = buildNodes(baselineYear, peakYear, neutralYear);

  return {
    baseline_year: baselineYear,
    peak_year: peakYear,
    neutral_year: neutralYear,
    assumptions,
    nodes,
  };
};
