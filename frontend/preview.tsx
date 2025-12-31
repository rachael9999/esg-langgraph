export type PreviewState = {
  svg: string;
  containerId: string;
};

export const renderPreview = ({ svg, containerId }: PreviewState): void => {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }
  container.innerHTML = svg;
};
