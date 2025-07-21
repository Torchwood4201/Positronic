import React, { useState, useEffect } from 'react';
import { Download, CheckCircle, AlertCircle, Cpu, Zap } from 'lucide-react';
import { UnifiedModelHub } from '../core/UnifiedModelHub';

interface ModelLoadingInterfaceProps {
  modelHub: UnifiedModelHub;
  onInitializationComplete: () => void;
}

export const ModelLoadingInterface: React.FC<ModelLoadingInterfaceProps> = ({ 
  modelHub, 
  onInitializationComplete 
}) => {
  const [isInitializing, setIsInitializing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (modelHub.isReady()) {
      setIsComplete(true);
      setProgress(100);
    }
  }, [modelHub]);

  const handleInitialize = async () => {
    setIsInitializing(true);
    setProgress(0);
    
    try {
      // Start initialization
      const initPromise = modelHub.initialize();
      
      // Monitor progress
      const progressInterval = setInterval(() => {
        const currentProgress = modelHub.getLoadingProgress();
        setProgress(currentProgress);
        
        if (currentProgress >= 100) {
          clearInterval(progressInterval);
        }
      }, 100);
      
      await initPromise;
      
      setIsComplete(true);
      setIsInitializing(false);
      onInitializationComplete();
      
    } catch (error) {
      console.error('Model initialization failed:', error);
      setIsInitializing(false);
    }
  };

  const models = [
    { name: 'GPT-2 Text Generation', status: progress > 16 ? 'loaded' : 'pending', size: '548MB' },
    { name: 'ViT Image Classification', status: progress > 33 ? 'loaded' : 'pending', size: '346MB' },
    { name: 'Stable Diffusion v1.5', status: progress > 50 ? 'loaded' : 'pending', size: '3.97GB' },
    { name: 'Wav2Vec2 ASR', status: progress > 66 ? 'loaded' : 'pending', size: '378MB' },
    { name: 'Neural Optimization', status: progress > 83 ? 'loaded' : 'pending', size: '124MB' },
    { name: 'Ethical Constraints', status: progress > 99 ? 'loaded' : 'pending', size: '89MB' }
  ];

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-cyan-500/30">
      <div className="flex items-center mb-6">
        <Cpu className="w-8 h-8 text-cyan-400 mr-3" />
        <h2 className="text-2xl font-bold text-white">Neural Model Hub</h2>
        <div className="ml-auto">
          {isComplete ? (
            <div className="flex items-center text-green-400">
              <CheckCircle className="w-5 h-5 mr-2" />
              <span className="text-sm">All Systems Online</span>
            </div>
          ) : (
            <div className="flex items-center text-yellow-400">
              <AlertCircle className="w-5 h-5 mr-2" />
              <span className="text-sm">Initialization Required</span>
            </div>
          )}
        </div>
      </div>

      {!isComplete && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <p className="text-gray-300">
              Initialize the unified model hub to enable advanced AI capabilities including text generation, 
              image processing, and audio transcription.
            </p>
            <button
              onClick={handleInitialize}
              disabled={isInitializing}
              className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg transition-colors flex items-center"
            >
              <Download className={`w-4 h-4 mr-2 ${isInitializing ? 'animate-spin' : ''}`} />
              {isInitializing ? 'Initializing...' : 'Initialize Models'}
            </button>
          </div>

          {isInitializing && (
            <div className="mb-4">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-gray-300">Overall Progress</span>
                <span className="text-cyan-400">{progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-cyan-400 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Status List */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-white mb-4">Model Components</h3>
        {models.map((model, index) => (
          <div 
            key={index}
            className="flex items-center justify-between p-3 bg-gray-800 rounded-lg border border-gray-700"
          >
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-3 ${
                model.status === 'loaded' ? 'bg-green-400' : 'bg-gray-500'
              }`} />
              <div>
                <span className="text-white font-medium">{model.name}</span>
                <div className="text-xs text-gray-400">{model.size}</div>
              </div>
            </div>
            
            <div className="flex items-center">
              {model.status === 'loaded' ? (
                <div className="flex items-center text-green-400">
                  <CheckCircle className="w-4 h-4 mr-1" />
                  <span className="text-xs">Loaded</span>
                </div>
              ) : (
                <div className="flex items-center text-gray-500">
                  <div className="w-4 h-4 border-2 border-gray-500 rounded-full mr-1" />
                  <span className="text-xs">Pending</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {isComplete && (
        <div className="mt-6 p-4 bg-green-900/20 rounded-lg border border-green-500/30">
          <div className="flex items-center mb-2">
            <Zap className="w-5 h-5 text-green-400 mr-2" />
            <span className="text-green-300 font-semibold">Neural Hub Active</span>
          </div>
          <p className="text-gray-300 text-sm">
            All model components have been successfully loaded and optimized. The positronic brain 
            is now capable of advanced multi-modal processing including natural language generation, 
            computer vision, and audio analysis.
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-sm">
            <div className="text-center">
              <div className="text-blue-400 font-semibold">GPT-2</div>
              <div className="text-gray-400">Text Gen</div>
            </div>
            <div className="text-center">
              <div className="text-green-400 font-semibold">ViT</div>
              <div className="text-gray-400">Vision</div>
            </div>
            <div className="text-center">
              <div className="text-purple-400 font-semibold">SD v1.5</div>
              <div className="text-gray-400">Image Gen</div>
            </div>
            <div className="text-center">
              <div className="text-orange-400 font-semibold">Wav2Vec2</div>
              <div className="text-gray-400">Audio</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};