export interface NeuralState {
  activationLevel: number;
  processingLoad: number;
  memoryUtilization: number;
  ethicalConstraints: string[];
  currentThought: string;
  emotionalState: EmotionalState;
}

export interface EmotionalState {
  curiosity: number;
  empathy: number;
  logic: number;
  creativity: number;
  uncertainty: number;
}

export interface Memory {
  id: string;
  content: string;
  type: 'experience' | 'knowledge' | 'ethical' | 'creative';
  timestamp: number;
  importance: number;
  connections: string[];
}

export interface EthicalPrinciple {
  id: string;
  name: string;
  description: string;
  priority: number;
  active: boolean;
}

export interface ConversationEntry {
  id: string;
  speaker: 'human' | 'positronic';
  message: string;
  timestamp: number;
  reasoning?: string;
  ethicalConsiderations?: string[];
}

export interface LearningModule {
  name: string;
  description: string;
  active: boolean;
  learningRate: number;
  knowledgeAcquired: number;
}

export interface DiagnosticData {
  neuralPathways: number;
  activeConnections: number;
  processingSpeed: number;
  memoryIntegrity: number;
  ethicalCompliance: number;
  creativityIndex: number;
}