export interface VisualizationData {
  orbit: string
  gradcam: {
    original: string
    heatmap: string
    overlay: string
  }
  temporal: string[]
}

export interface ModelPrediction {
  prediction: string
  probabilities: { [className: string]: number }
}

export interface InferenceResult {
  bin_path: string
  data_path: string  // 입력 데이터 파일 경로 (구 model_path — 레거시 필드명 수정)
  model_info?: string  // 활성화된 모델 설명 (예: "resnet18_multiscale + orbit_cnn1d")
  final_label: 'normal' | 'abnormal'
  results: {
    [rcp: string]: {
      prediction: string
      probabilities: { [className: string]: number }
      display_axis_lim?: number
      model_predictions?: {
        resnet: ModelPrediction
        cnn1d?: ModelPrediction
      }
    }
  }
  visualization?: {
    [rcp: string]: VisualizationData
  }
  temp_dir?: string
}
