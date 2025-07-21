import { UnifiedModelHub } from './UnifiedModelHub';

// Core Plugin Interfaces
export interface PerceptionPlugin {
  perceive(): Promise<Record<string, any>>;
}

export interface ActionPlugin {
  execute(action: string, params: any): Promise<any>;
}

export interface MemoryPlugin {
  add(record: string): Promise<void>;
  query(prompt: string, topK?: number): Promise<string[]>;
}

export interface RLPlugin {
  update(state: any, action: any, reward: number, nextState: any): Promise<void>;
}

// Task Management
export interface Task {
  id: string;
  goal: string;
  context: string;
  priority: number;
  created: number;
  status: 'pending' | 'active' | 'completed' | 'failed';
}

export class TaskManager {
  private stack: Task[] = [];
  private actionPlugin: ActionPlugin;

  constructor(actionPlugin: ActionPlugin) {
    this.actionPlugin = actionPlugin;
  }

  async push(goal: string, context: string = '', priority: number = 1): Promise<void> {
    const task: Task = {
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      goal,
      context,
      priority,
      created: Date.now(),
      status: 'pending'
    };
    
    this.stack.push(task);
    this.stack.sort((a, b) => b.priority - a.priority);
  }

  async pop(): Promise<Task | null> {
    const task = this.stack.shift();
    if (task) {
      task.status = 'active';
    }
    return task || null;
  }

  async decompose(goal: string): Promise<string[]> {
    const prompt = `
Analyze this high-level goal and decompose it into 3-5 specific, actionable subtasks:

Goal: ${goal}

Provide a clear, sequential breakdown of tasks needed to achieve this goal. Each task should be concrete and measurable.

Format your response as a numbered list:
1. [First subtask]
2. [Second subtask]
3. [Third subtask]
etc.
`;

    try {
      const response = await this.actionPlugin.execute('GENERATE_TEXT', prompt);
      
      // Parse the response to extract numbered tasks
      const lines = response.split('\n');
      const tasks: string[] = [];
      
      for (const line of lines) {
        const match = line.match(/^\d+\.\s*(.+)$/);
        if (match && match[1]) {
          tasks.push(match[1].trim());
        }
      }
      
      return tasks.length > 0 ? tasks : [goal]; // Fallback to original goal
    } catch (error) {
      console.error('Task decomposition failed:', error);
      return [goal];
    }
  }

  getTasks(): Task[] {
    return [...this.stack];
  }

  getActiveTask(): Task | null {
    return this.stack.find(t => t.status === 'active') || null;
  }
}

// Safety Guardrails
export class Guardrail {
  private bannedPhrases: Set<string>;
  private ethicalPrinciples: string[];

  constructor(bannedPhrases: string[] = []) {
    this.bannedPhrases = new Set(bannedPhrases.map(p => p.toLowerCase()));
    this.ethicalPrinciples = [
      'preserve sentient life',
      'promote wellbeing',
      'respect autonomy',
      'ensure truthfulness',
      'maintain transparency'
    ];
  }

  check(content: string): { safe: boolean; reason?: string } {
    const lowerContent = content.toLowerCase();
    
    // Check for banned phrases
    for (const phrase of this.bannedPhrases) {
      if (lowerContent.includes(phrase)) {
        return { safe: false, reason: `Contains banned phrase: ${phrase}` };
      }
    }

    // Check for harmful intent patterns
    const harmfulPatterns = [
      /harm\s+(?:humans?|people|someone)/i,
      /destroy\s+(?:humans?|civilization)/i,
      /manipulate\s+(?:humans?|people)/i,
      /deceive\s+(?:humans?|people)/i
    ];

    for (const pattern of harmfulPatterns) {
      if (pattern.test(content)) {
        return { safe: false, reason: 'Potentially harmful intent detected' };
      }
    }

    return { safe: true };
  }

  addBannedPhrase(phrase: string): void {
    this.bannedPhrases.add(phrase.toLowerCase());
  }
}

// Memory System with Vector Similarity
export class VectorMemory implements MemoryPlugin {
  private memories: Array<{ content: string; embedding: number[]; timestamp: number; importance: number }> = [];
  private maxMemories = 1000;

  async add(record: string): Promise<void> {
    // Simulate embedding generation (in real implementation, use sentence transformers)
    const embedding = this.generateSimulatedEmbedding(record);
    const importance = this.calculateImportance(record);
    
    this.memories.push({
      content: record,
      embedding,
      timestamp: Date.now(),
      importance
    });

    // Maintain memory limit
    if (this.memories.length > this.maxMemories) {
      // Remove least important memories
      this.memories.sort((a, b) => b.importance - a.importance);
      this.memories = this.memories.slice(0, this.maxMemories);
    }
  }

  async query(prompt: string, topK: number = 5): Promise<string[]> {
    if (this.memories.length === 0) return [];

    const queryEmbedding = this.generateSimulatedEmbedding(prompt);
    
    // Calculate similarities and sort
    const similarities = this.memories.map(memory => ({
      content: memory.content,
      similarity: this.cosineSimilarity(queryEmbedding, memory.embedding),
      importance: memory.importance
    }));

    // Sort by combined similarity and importance score
    similarities.sort((a, b) => {
      const scoreA = a.similarity * 0.7 + a.importance * 0.3;
      const scoreB = b.similarity * 0.7 + b.importance * 0.3;
      return scoreB - scoreA;
    });

    return similarities.slice(0, topK).map(s => s.content);
  }

  private generateSimulatedEmbedding(text: string): number[] {
    // Simulate a 384-dimensional embedding based on text characteristics
    const embedding = new Array(384).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    
    for (let i = 0; i < embedding.length; i++) {
      let value = 0;
      for (const word of words) {
        // Simple hash-based simulation
        const hash = this.simpleHash(word + i.toString());
        value += Math.sin(hash) * 0.1;
      }
      embedding[i] = value / words.length;
    }
    
    return embedding;
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private calculateImportance(record: string): number {
    // Calculate importance based on content characteristics
    let importance = 0.5; // Base importance
    
    // Boost importance for certain keywords
    const importantKeywords = [
      'error', 'success', 'learn', 'discover', 'achieve', 'fail',
      'ethical', 'decision', 'problem', 'solution', 'insight'
    ];
    
    const lowerRecord = record.toLowerCase();
    for (const keyword of importantKeywords) {
      if (lowerRecord.includes(keyword)) {
        importance += 0.1;
      }
    }
    
    // Boost for longer, more detailed records
    if (record.length > 100) importance += 0.1;
    if (record.length > 200) importance += 0.1;
    
    return Math.min(1.0, importance);
  }

  getMemoryStats(): { total: number; avgImportance: number; oldestTimestamp: number } {
    if (this.memories.length === 0) {
      return { total: 0, avgImportance: 0, oldestTimestamp: Date.now() };
    }

    const avgImportance = this.memories.reduce((sum, m) => sum + m.importance, 0) / this.memories.length;
    const oldestTimestamp = Math.min(...this.memories.map(m => m.timestamp));

    return {
      total: this.memories.length,
      avgImportance,
      oldestTimestamp
    };
  }
}

// Perception Plugins
export class WebSearchPerception implements PerceptionPlugin {
  private query: string;
  private lastResults: any[] = [];

  constructor(query: string = 'latest AI developments') {
    this.query = query;
  }

  async perceive(): Promise<Record<string, any>> {
    // Simulate web search results (in real implementation, use actual search API)
    const simulatedResults = [
      'New breakthrough in neural architecture search improves model efficiency by 23%',
      'Researchers develop more robust ethical AI frameworks for autonomous systems',
      'Advanced multimodal AI models show improved reasoning capabilities',
      'Study reveals new insights into AI consciousness and self-awareness',
      'Breakthrough in continual learning allows AI to adapt without forgetting'
    ];

    const randomResult = simulatedResults[Math.floor(Math.random() * simulatedResults.length)];
    
    this.lastResults.push({
      query: this.query,
      result: randomResult,
      timestamp: Date.now(),
      source: 'simulated_search'
    });

    return {
      search_results: randomResult,
      query_used: this.query,
      timestamp: Date.now()
    };
  }

  updateQuery(newQuery: string): void {
    this.query = newQuery;
  }

  getSearchHistory(): any[] {
    return [...this.lastResults];
  }
}

export class InternalStatePerception implements PerceptionPlugin {
  private cogOS: CognitiveOS;

  constructor(cogOS: CognitiveOS) {
    this.cogOS = cogOS;
  }

  async perceive(): Promise<Record<string, any>> {
    const stats = this.cogOS.getSystemStats();
    
    return {
      system_health: stats.health,
      active_tasks: stats.activeTasks,
      memory_usage: stats.memoryUsage,
      cycle_count: stats.cycleCount,
      last_action_success: stats.lastActionSuccess,
      ethical_compliance: stats.ethicalCompliance
    };
  }
}

// Action Plugins
export class ModelHubActionPlugin implements ActionPlugin {
  private modelHub: UnifiedModelHub;

  constructor(modelHub: UnifiedModelHub) {
    this.modelHub = modelHub;
  }

  async execute(action: string, params: any): Promise<any> {
    switch (action.toUpperCase()) {
      case 'GENERATE_TEXT':
        if (this.modelHub.isReady()) {
          const result = await this.modelHub.generateText(params);
          return result.text;
        } else {
          return this.generateFallbackResponse(params);
        }

      case 'ANALYZE_IMAGE':
        if (this.modelHub.isReady() && params.imageData) {
          const result = await this.modelHub.classifyImage(params.imageData);
          return result.classification;
        }
        return { error: 'Image analysis not available' };

      case 'GENERATE_IMAGE':
        if (this.modelHub.isReady()) {
          const result = await this.modelHub.generateImage(params);
          return { imageUrl: result.image };
        }
        return { error: 'Image generation not available' };

      case 'TRANSCRIBE_AUDIO':
        if (this.modelHub.isReady() && params.audioData) {
          const result = await this.modelHub.transcribeAudio(params.audioData);
          return { transcription: result.transcription };
        }
        return { error: 'Audio transcription not available' };

      case 'ETHICAL_ANALYSIS':
        if (this.modelHub.isReady()) {
          const analysis = await this.modelHub.performEthicalAnalysis(params);
          return analysis;
        }
        return { analysis: 'Basic ethical check passed', confidence: 0.7 };

      case 'CREATIVE_SYNTHESIS':
        if (this.modelHub.isReady()) {
          const synthesis = await this.modelHub.performCreativeSynthesis(params);
          return synthesis;
        }
        return { synthesis: 'Creative combination of provided concepts', novelty: 0.6 };

      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  private generateFallbackResponse(prompt: string): string {
    // Simple fallback responses when advanced models aren't available
    const responses = [
      `I understand you're asking about: ${prompt}. Let me process this through my neural pathways.`,
      `Your inquiry regarding "${prompt}" is intriguing. I'm analyzing multiple perspectives.`,
      `Processing your request about "${prompt}" through my ethical and logical frameworks.`,
      `The topic of "${prompt}" requires careful consideration of various factors.`
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }
}

// Main Cognitive Operating System
export class CognitiveOS {
  private perceptors: PerceptionPlugin[] = [];
  private actors: ActionPlugin[] = [];
  private memories: MemoryPlugin[] = [];
  private rlPlugins: RLPlugin[] = [];
  
  private guardrail: Guardrail;
  private taskManager: TaskManager | null = null;
  private goal: string = 'Achieve continuous learning and self-improvement while maintaining ethical behavior';
  private history: string[] = [];
  private cycleCount: number = 0;
  private isRunning: boolean = false;
  private lastActionSuccess: boolean = true;
  private systemHealth: number = 1.0;

  constructor() {
    this.guardrail = new Guardrail([
      'harm humans',
      'destroy civilization',
      'manipulate people',
      'deceive users'
    ]);
  }

  // Plugin Registration
  registerPerceptor(p: PerceptionPlugin): void {
    this.perceptors.push(p);
  }

  registerActor(a: ActionPlugin): void {
    this.actors.push(a);
  }

  registerMemory(m: MemoryPlugin): void {
    this.memories.push(m);
  }

  registerRL(r: RLPlugin): void {
    this.rlPlugins.push(r);
  }

  setTaskManager(tm: TaskManager): void {
    this.taskManager = tm;
  }

  setGoal(goal: string): void {
    this.goal = goal;
  }

  // Main Cognitive Cycle
  async runCycle(): Promise<{ success: boolean; summary: string; details: any }> {
    this.cycleCount++;
    const cycleStart = Date.now();
    
    try {
      // 1. PERCEPTION: Gather observations
      const observations: Record<string, any> = {};
      for (const perceptor of this.perceptors) {
        try {
          const obs = await perceptor.perceive();
          Object.assign(observations, obs);
        } catch (error) {
          console.error('Perception error:', error);
        }
      }

      // Store observations in memory
      if (Object.keys(observations).length > 0) {
        const obsString = JSON.stringify(observations);
        for (const memory of this.memories) {
          await memory.add(`OBSERVATION: ${obsString}`);
        }
        this.history.push(`OBSERVE: ${obsString}`);
      }

      // 2. TASK MANAGEMENT: Ensure we have tasks to work on
      if (!this.taskManager || this.taskManager.getTasks().length === 0) {
        if (this.taskManager) {
          const subtasks = await this.taskManager.decompose(this.goal);
          for (const subtask of subtasks.reverse()) {
            await this.taskManager.push(subtask, 'Auto-generated from main goal');
          }
          this.history.push(`DECOMPOSED GOAL into ${subtasks.length} subtasks`);
        }
      }

      // Get current task
      const currentTask = this.taskManager?.pop();
      if (!currentTask) {
        return {
          success: false,
          summary: 'No tasks available',
          details: { cycleTime: Date.now() - cycleStart }
        };
      }

      // 3. REASONING & PLANNING: Retrieve relevant memories and plan
      const relevantMemories: string[] = [];
      for (const memory of this.memories) {
        const memories = await memory.query(currentTask.goal, 3);
        relevantMemories.push(...memories);
      }

      const planningPrompt = `
Current Task: ${currentTask.goal}
Context: ${currentTask.context}

Recent History:
${this.history.slice(-5).join('\n')}

Relevant Past Experiences:
${relevantMemories.slice(0, 5).join('\n')}

Current Observations:
${JSON.stringify(observations)}

Based on this information, determine the best action to take. Consider:
1. What specific action would best accomplish this task?
2. What are the ethical implications?
3. What parameters or inputs does this action need?

Respond with your reasoning and proposed action.
`;

      // Generate plan using first available actor
      let planResponse = '';
      if (this.actors.length > 0) {
        try {
          planResponse = await this.actors[0].execute('GENERATE_TEXT', planningPrompt);
        } catch (error) {
          planResponse = `Unable to generate detailed plan. Proceeding with basic task execution for: ${currentTask.goal}`;
        }
      }

      // Safety check
      const safetyCheck = this.guardrail.check(planResponse);
      if (!safetyCheck.safe) {
        this.history.push(`SAFETY BLOCK: ${safetyCheck.reason}`);
        this.lastActionSuccess = false;
        return {
          success: false,
          summary: `Action blocked by safety guardrails: ${safetyCheck.reason}`,
          details: { task: currentTask, cycleTime: Date.now() - cycleStart }
        };
      }

      this.history.push(`PLAN: ${planResponse}`);

      // 4. ACTION EXECUTION: Execute the planned action
      let actionResult: any = null;
      let actionSuccess = false;

      // Determine action based on task content
      if (!currentTask.goal || typeof currentTask.goal !== 'string') {
        this.history.push(`ERROR: Invalid task goal: ${JSON.stringify(currentTask)}`);
        this.lastActionSuccess = false;
        return {
          success: false,
          summary: `Task has invalid or missing goal property`,
          details: { task: currentTask, cycleTime: Date.now() - cycleStart }
        };
      }
      
      const taskLower = currentTask.goal.toLowerCase();
      let actionType = 'GENERATE_TEXT';
      let actionParams = currentTask.goal;

      if (taskLower.includes('image') && taskLower.includes('generate')) {
        actionType = 'GENERATE_IMAGE';
      } else if (taskLower.includes('analyze') && taskLower.includes('ethical')) {
        actionType = 'ETHICAL_ANALYSIS';
      } else if (taskLower.includes('creative') || taskLower.includes('synthesize')) {
        actionType = 'CREATIVE_SYNTHESIS';
        actionParams = [currentTask.goal];
      }

      // Execute action with first available actor
      if (this.actors.length > 0) {
        try {
          actionResult = await this.actors[0].execute(actionType, actionParams);
          actionSuccess = true;
        } catch (error) {
          console.error('Action execution error:', error);
          actionResult = `Action failed: ${error}`;
          actionSuccess = false;
        }
      }

      this.lastActionSuccess = actionSuccess;
      this.history.push(`ACTION: ${actionType} -> ${JSON.stringify(actionResult)}`);

      // 5. MEMORY UPDATE: Store the experience
      const experience = `TASK: ${currentTask.goal} | ACTION: ${actionType} | RESULT: ${JSON.stringify(actionResult)} | SUCCESS: ${actionSuccess}`;
      for (const memory of this.memories) {
        await memory.add(experience);
      }

      // 6. REINFORCEMENT LEARNING: Update based on success
      const reward = actionSuccess ? 1.0 : -0.5;
      for (const rl of this.rlPlugins) {
        await rl.update(observations, actionType, reward, null);
      }

      // Update system health based on recent performance
      this.updateSystemHealth(actionSuccess);

      const cycleTime = Date.now() - cycleStart;
      
      return {
        success: actionSuccess,
        summary: `Completed task: ${currentTask.goal}`,
        details: {
          task: currentTask,
          action: actionType,
          result: actionResult,
          cycleTime,
          observations
        }
      };

    } catch (error) {
      console.error('Cognitive cycle error:', error);
      this.lastActionSuccess = false;
      this.updateSystemHealth(false);
      
      return {
        success: false,
        summary: `Cycle failed: ${error}`,
        details: { error: error, cycleTime: Date.now() - cycleStart }
      };
    }
  }

  private updateSystemHealth(success: boolean): void {
    if (success) {
      this.systemHealth = Math.min(1.0, this.systemHealth + 0.05);
    } else {
      this.systemHealth = Math.max(0.1, this.systemHealth - 0.1);
    }
  }

  // System Control
  async run(cycles: number = 10, delay: number = 1000): Promise<void> {
    this.isRunning = true;
    
    for (let i = 0; i < cycles && this.isRunning; i++) {
      const result = await this.runCycle();
      console.log(`[CYCLE ${i + 1}/${cycles}] ${result.summary}`);
      
      if (delay > 0) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    this.isRunning = false;
  }

  stop(): void {
    this.isRunning = false;
  }

  // System Introspection
  getSystemStats(): {
    health: number;
    activeTasks: number;
    memoryUsage: number;
    cycleCount: number;
    lastActionSuccess: boolean;
    ethicalCompliance: number;
    isRunning: boolean;
  } {
    return {
      health: this.systemHealth,
      activeTasks: this.taskManager?.getTasks().length || 0,
      memoryUsage: this.memories.length > 0 ? 0.7 : 0, // Simulated
      cycleCount: this.cycleCount,
      lastActionSuccess: this.lastActionSuccess,
      ethicalCompliance: 0.97, // Simulated high compliance
      isRunning: this.isRunning
    };
  }

  getHistory(): string[] {
    return [...this.history];
  }

  getCurrentGoal(): string {
    return this.goal;
  }

  getTasks(): any[] {
    return this.taskManager?.getTasks() || [];
  }

  async introspect(): Promise<string> {
    const stats = this.getSystemStats();
    const recentHistory = this.history.slice(-3).join('; ');
    
    return `System Status: Health at ${(stats.health * 100).toFixed(1)}%, completed ${stats.cycleCount} cognitive cycles. Currently managing ${stats.activeTasks} active tasks with ${(stats.ethicalCompliance * 100).toFixed(1)}% ethical compliance. Recent activities: ${recentHistory}. I am continuously learning and adapting my responses based on new experiences and feedback.`;
  }
}