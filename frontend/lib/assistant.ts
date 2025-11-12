import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

export async function getStylingSuggestion(garmentName: string, audience?: string, context?: string): Promise<string> {
  const { data } = await axios.post<{ suggestion: string }>(`${API_BASE}/assistant/suggest`, {
    garment_name: garmentName,
    audience,
    context
  });
  return data.suggestion;
}

