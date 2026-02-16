
import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { 
  GoogleGenAI, 
  Modality, 
  GenerateContentResponse,
  Type
} from "@google/genai";

// --- Types & Constants ---
type ToolType = 'think' | 'imagine' | 'speak' | 'animate' | 'analyze';

interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

interface GeneratedMedia {
  type: 'image' | 'video' | 'audio' | 'analysis';
  url?: string;
  prompt: string;
  result?: string;
  timestamp: number;
}

// --- Utils ---
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Icons ---
const Icons = {
  Think: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/><path d="M19 3v4"/><path d="M21 5h-4"/></svg>
  ),
  Imagine: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
  ),
  Speak: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
  ),
  Animate: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="20" height="14" x="2" y="3" rx="2"/><path d="M22 7l-3.5 2"/><path d="M22 13l-3.5-2"/><path d="M2 11h16.5"/></svg>
  ),
  Analyze: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21 21-4.3-4.3"/><circle cx="10" cy="10" r="7"/><path d="M10 7v6"/><path d="M7 10h6"/></svg>
  ),
  Send: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
  ),
  Loader: () => (
    <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
  ),
  Alert: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
  )
};

// --- App Component ---
const OmniStudio = () => {
  const [activeTool, setActiveTool] = useState<ToolType>('think');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [mediaGallery, setMediaGallery] = useState<GeneratedMedia[]>([]);
  const [videoStatus, setVideoStatus] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const initAudio = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    }
  };

  const handleError = async (err: any) => {
    console.error(err);
    const msg = err.message || "An unexpected error occurred.";
    
    if (msg.includes("Requested entity was not found")) {
      setError("API session expired. Re-selecting key...");
      await (window as any).aistudio.openSelectKey();
      setTimeout(() => setError(null), 3000);
    } else {
      setError(msg);
      setTimeout(() => setError(null), 5000);
    }
  };

  const handleGenerate = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError(null);
    initAudio();

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      if (activeTool === 'think') {
        const response = await ai.models.generateContent({
          model: 'gemini-3-pro-preview',
          contents: { parts: [{ text: input }] },
          config: { thinkingConfig: { thinkingBudget: 32768 } }
        });
        setChatHistory(prev => [...prev, { role: 'user', text: input }, { role: 'model', text: response.text || 'No response.' }]);
        setInput('');
      } 
      
      else if (activeTool === 'analyze') {
        const response = await ai.models.generateContent({
          model: 'gemini-3-pro-preview',
          contents: { parts: [{ text: `Act as a professional data screener. Analyze the following input and provide a structured summary including risks, key points, and overall sentiment: \n\n${input}` }] },
          config: { thinkingConfig: { thinkingBudget: 16000 } }
        });
        setMediaGallery(prev => [{ type: 'analysis', result: response.text, prompt: input, timestamp: Date.now() }, ...prev]);
        setInput('');
      }

      else if (activeTool === 'imagine') {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: { parts: [{ text: input }] },
          config: { imageConfig: { aspectRatio: "16:9" } }
        });

        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            const url = `data:image/png;base64,${part.inlineData.data}`;
            setMediaGallery(prev => [{ type: 'image', url, prompt: input, timestamp: Date.now() }, ...prev]);
          }
        }
      } 
      
      else if (activeTool === 'speak') {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash-preview-tts',
          contents: [{ parts: [{ text: input }] }],
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } }
          }
        });

        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (base64Audio && audioContextRef.current) {
          const audioBuffer = await decodeAudioData(decode(base64Audio), audioContextRef.current, 24000, 1);
          const source = audioContextRef.current.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioContextRef.current.destination);
          source.start();
          setMediaGallery(prev => [{ type: 'audio', prompt: input, timestamp: Date.now() }, ...prev]);
        }
      } 
      
      else if (activeTool === 'animate') {
        const hasKey = await (window as any).aistudio.hasSelectedApiKey();
        if (!hasKey) await (window as any).aistudio.openSelectKey();

        const veoAi = new GoogleGenAI({ apiKey: process.env.API_KEY });
        setVideoStatus("Calibrating cinematic engine...");
        
        let operation = await veoAi.models.generateVideos({
          model: 'veo-3.1-fast-generate-preview',
          prompt: input,
          config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
        });

        while (!operation.done) {
          setVideoStatus("Rendering frames... usually takes 45-90 seconds");
          await new Promise(resolve => setTimeout(resolve, 8000));
          operation = await veoAi.operations.getVideosOperation({ operation: operation });
        }

        const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
        const videoResponse = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
        if (!videoResponse.ok) throw new Error("Failed to download generated video.");
        
        const videoBlob = await videoResponse.blob();
        const videoUrl = URL.createObjectURL(videoBlob);

        setMediaGallery(prev => [{ type: 'video', url: videoUrl, prompt: input, timestamp: Date.now() }, ...prev]);
        setVideoStatus(null);
      }
    } catch (err: any) {
      handleError(err);
    } finally {
      setLoading(false);
      setVideoStatus(null);
    }
  };

  return (
    <div className="flex h-full w-full bg-[#0a0a0a] text-gray-200 overflow-hidden selection:bg-indigo-500/30">
      {/* Sidebar Navigation */}
      <aside className="w-16 md:w-20 border-r border-white/5 flex flex-col items-center py-8 space-y-8 glass z-30">
        <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center neo-shadow">
          <span className="font-bold text-lg text-white">Ω</span>
        </div>
        
        <nav className="flex flex-col space-y-6">
          <NavButton active={activeTool === 'think'} onClick={() => setActiveTool('think')} icon={<Icons.Think />} label="Think" />
          <NavButton active={activeTool === 'analyze'} onClick={() => setActiveTool('analyze')} icon={<Icons.Analyze />} label="Analyze" />
          <NavButton active={activeTool === 'imagine'} onClick={() => setActiveTool('imagine')} icon={<Icons.Imagine />} label="Imagine" />
          <NavButton active={activeTool === 'speak'} onClick={() => setActiveTool('speak')} icon={<Icons.Speak />} label="Speak" />
          <NavButton active={activeTool === 'animate'} onClick={() => setActiveTool('animate')} icon={<Icons.Animate />} label="Animate" />
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Error Banner */}
        {error && (
          <div className="absolute top-0 left-0 right-0 z-50 p-4 bg-red-500/90 backdrop-blur-md text-white flex items-center justify-between animate-in slide-in-from-top duration-300">
            <div className="flex items-center space-x-3">
              <Icons.Alert />
              <span className="text-sm font-medium">{error}</span>
            </div>
            <button onClick={() => setError(null)} className="text-white/60 hover:text-white"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg></button>
          </div>
        )}

        {/* Header */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-6 glass shrink-0 z-20">
          <div className="flex items-center space-x-3">
            <h1 className="text-lg font-semibold tracking-tight text-white capitalize">{activeTool} Workspace</h1>
            <div className="flex items-center space-x-2">
               <div className={`w-2 h-2 rounded-full ${loading ? 'bg-indigo-500 animate-pulse' : 'bg-green-500/40'}`}></div>
               <span className="text-[10px] text-gray-500 uppercase tracking-widest">{loading ? 'Active' : 'Standby'}</span>
            </div>
          </div>
          <div className="flex items-center space-x-4">
             <button onClick={() => (window as any).aistudio.openSelectKey()} className="text-[10px] bg-white/5 hover:bg-white/10 px-3 py-1.5 rounded-full border border-white/5 transition-colors">Switch Key</button>
          </div>
        </header>

        {/* Workspace Views */}
        <div className="flex-1 flex overflow-hidden">
          <section className="flex-1 flex flex-col p-6 overflow-y-auto relative bg-[#0d0d0d]">
            {activeTool === 'think' ? (
              <div className="flex-1 space-y-6 max-w-4xl mx-auto w-full pb-32">
                {chatHistory.length === 0 && (
                  <WelcomeState icon={<Icons.Think />} title="Brainstorming Chamber" subtitle="Leveraging Gemini 3 Pro for deep reasoning and complex problem solving." />
                )}
                {chatHistory.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
                    <div className={`max-w-[85%] p-5 rounded-2xl shadow-xl ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white/5 border border-white/10'}`}>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            ) : activeTool === 'analyze' ? (
               <div className="flex-1 space-y-6 max-w-5xl mx-auto w-full pb-32">
                  {mediaGallery.filter(m => m.type === 'analysis').length === 0 && (
                    <WelcomeState icon={<Icons.Analyze />} title="Screening Laboratory" subtitle="Replace your manual screening scripts with automated AI data analysis." />
                  )}
                  {mediaGallery.filter(m => m.type === 'analysis').map((item, i) => (
                    <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-6 font-mono text-xs leading-relaxed animate-in zoom-in-95 duration-500">
                      <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-2">
                        <span className="text-indigo-400">ANALYSIS_REPORT_0{i}</span>
                        <span className="text-gray-600 italic">Target: {item.prompt.substring(0, 30)}...</span>
                      </div>
                      <pre className="whitespace-pre-wrap text-gray-300">{item.result}</pre>
                    </div>
                  ))}
               </div>
            ) : (
              <div className="flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-32 max-w-7xl mx-auto w-full">
                {mediaGallery.filter(m => 
                  (activeTool === 'imagine' && m.type === 'image') || 
                  (activeTool === 'animate' && m.type === 'video') ||
                  (activeTool === 'speak' && m.type === 'audio')
                ).length === 0 && (
                  <div className="col-span-full h-full flex items-center justify-center">
                    <WelcomeState icon={activeTool === 'imagine' ? <Icons.Imagine /> : activeTool === 'animate' ? <Icons.Animate /> : <Icons.Speak />} title={`Create with ${activeTool}`} subtitle="Your creative outputs will be rendered here in high fidelity." />
                  </div>
                )}
                {mediaGallery
                  .filter(m => (activeTool === 'imagine' && m.type === 'image') || (activeTool === 'animate' && m.type === 'video') || (activeTool === 'speak' && m.type === 'audio'))
                  .map((item, i) => (
                  <div key={i} className="group relative bg-white/5 border border-white/10 rounded-2xl overflow-hidden neo-shadow hover:border-indigo-500/30 transition-all h-fit animate-in zoom-in-95">
                    {item.type === 'image' && <img src={item.url} className="w-full aspect-video object-cover" alt={item.prompt} />}
                    {item.type === 'video' && <video src={item.url} controls className="w-full aspect-video object-cover" />}
                    {item.type === 'audio' && (
                      <div className="p-8 flex flex-col items-center justify-center space-y-4 bg-indigo-900/10">
                        <div className="w-12 h-12 bg-indigo-500/20 rounded-full flex items-center justify-center text-indigo-400"><Icons.Speak /></div>
                        <span className="text-[10px] text-gray-500 font-mono">SYNTHESIZED_VOICE.RAW</span>
                      </div>
                    )}
                    <div className="p-4 bg-black/40 backdrop-blur-sm">
                      <p className="text-[11px] text-gray-400 line-clamp-2 leading-relaxed italic">"{item.prompt}"</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Input Container */}
            <div className="absolute bottom-8 left-0 right-0 flex justify-center px-6 pointer-events-none z-40">
              <div className="max-w-4xl w-full pointer-events-auto">
                {videoStatus && (
                  <div className="bg-indigo-950/80 border border-indigo-500/50 rounded-lg p-3 text-[11px] text-indigo-300 flex items-center justify-between mb-3 backdrop-blur-lg">
                    <div className="flex items-center space-x-2">
                      <Icons.Loader />
                      <span className="font-medium tracking-wide">{videoStatus}</span>
                    </div>
                  </div>
                )}
                <div className="flex items-center bg-[#151515]/90 border border-white/10 rounded-2xl p-2 shadow-2xl focus-within:border-indigo-500/50 backdrop-blur-xl transition-all">
                  <textarea
                    rows={1}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleGenerate())}
                    placeholder={activeTool === 'analyze' ? "Paste data to analyze..." : `Input for ${activeTool}...`}
                    className="flex-1