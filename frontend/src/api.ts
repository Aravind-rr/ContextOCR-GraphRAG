import type {DocumentInfo,EvaluationReport,Run,SystemStatus} from './types'
export const API=import.meta.env.VITE_API_URL||'http://localhost:8000'
async function request<T>(path:string, init?:RequestInit):Promise<T>{const res=await fetch(`${API}${path}`,init);if(!res.ok){let msg=`Request failed (${res.status})`;try{const body=await res.json();msg=body.detail||msg}catch{}throw new Error(msg)}return res.json() as Promise<T>}
export const api={
 status:()=>request<SystemStatus>('/api/system/status'),
 history:()=>request<Run[]>('/api/runs'),
 run:(id:string)=>request<Run>(`/api/runs/${id}`),
 upload:(file:File)=>{const form=new FormData();form.append('file',file);return request<DocumentInfo>('/api/documents/upload',{method:'POST',body:form})},
 demo:()=>request<DocumentInfo>('/api/documents/demo',{method:'POST'}),
 start:(document_id:string,confidence_threshold:number,domain:string)=>request<Run>('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_id,confidence_threshold,domain,start_stage:1})}),
 evaluate:(run_id:string,ground_truth?:string)=>request<EvaluationReport>('/api/evaluation/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id,ground_truth:ground_truth?.trim()||null})}),
 evaluateDataset:()=>request<EvaluationReport>('/api/evaluation/dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_ids:[],ground_truths:{},quality_labels:{}})}),
 evaluationChart:(evaluation_id:string,name:string)=>`${API}/api/evaluation/${evaluation_id}/charts/${name}`,
 experiments:(run_id:string,ground_truth:string)=>request<{experiments:Array<{name:string;status:string;metrics:Record<string,number>|null;reason?:string}>}>('/api/experiments/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id,ground_truth})})
}
