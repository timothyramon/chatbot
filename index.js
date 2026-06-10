import express from 'express';
import { Anthropic } from '@anthropic-ai/sdk';
import { createClient } from '@supabase/supabase-js';
import multer from 'multer';
import cors from 'cors';
import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import fs from 'fs';

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json());

// Initialize clients
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Helper: Extract text from various file types
async function extractText(fileBuffer, filename) {
  if (filename.endsWith('.pdf')) {
    const data = await pdfParse(fileBuffer);
    return data.text;
  } else if (filename.endsWith('.docx')) {
    const result = await mammoth.extractRawText({ buffer: fileBuffer });
    return result.value;
  } else {
    return fileBuffer.toString('utf-8');
  }
}

// Helper: Get embedding from Claude
async function getEmbedding(text) {
  const response = await anthropic.embeddings.create({
    model: "claude-3-haiku-20240307",
    input: text.slice(0, 100000) // Claude limit
  });
  return response.embeddings[0];
}

// UPLOAD DOCUMENT ENDPOINT
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    const { org_id, user_email } = req.body;
    const file = req.file;
    
    // Extract text from document
    const text = await extractText(file.buffer, file.originalname);
    
    // Generate embedding
    const embedding = await getEmbedding(text);
    
    // Upload to Supabase storage
    const filePath = `${org_id}/${Date.now()}_${file.originalname}`;
    await supabase.storage.from('documents').upload(filePath, file.buffer);
    
    // Store in database with vector
    const { data, error } = await supabase
      .from('documents')
      .insert({
        org_id: org_id,
        uploaded_by: user_email,
        filename: file.originalname,
        file_path: filePath,
        content: text.slice(0, 50000), // Store first 50k chars
        embedding: embedding
      })
      .select();
    
    if (error) throw error;
    
    res.json({ success: true, document: data[0] });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// QUERY ENDPOINT
app.post('/api/query', async (req, res) => {
  try {
    const { question, org_id } = req.body;
    
    // Get embedding for question
    const questionEmbedding = await getEmbedding(question);
    
    // Find relevant documents
    const { data: documents } = await supabase.rpc('match_documents', {
      query_embedding: questionEmbedding,
      match_threshold: 0.7,
      match_count: 5,
      p_org_id: org_id
    });
    
    if (!documents || documents.length === 0) {
      return res.json({
        answer: "No documents found. Please upload some documents first.",
        sources: []
      });
    }
    
    // Prepare context for Claude
    const context = documents.map(doc => 
      `[Document: ${doc.filename}]\n${doc.content}\n`
    ).join('\n');
    
    // Ask Claude
    const response = await anthropic.messages.create({
      model: "claude-3-sonnet-20241022",
      max_tokens: 2000,
      temperature: 0.3,
      system: "You are a document assistant. ONLY answer using the provided documents. If information isn't in the documents, say 'This information is not available in uploaded documents.' Always cite the source document name.",
      messages: [{
        role: "user",
        content: `Documents:\n${context}\n\nQuestion: ${question}\n\nAnswer based ONLY on the documents above:`
      }]
    });
    
    // Log the query
    await supabase.from('queries').insert({
      org_id: org_id,
      question: question,
      answer: response.content[0].text,
      sources: documents.map(d => d.filename)
    });
    
    res.json({
      answer: response.content[0].text,
      sources: documents.map(d => d.filename)
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// LIST DOCUMENTS ENDPOINT
app.get('/api/documents/:org_id', async (req, res) => {
  const { data, error } = await supabase
    .from('documents')
    .select('id, filename, uploaded_at')
    .eq('org_id', req.params.org_id)
    .order('uploaded_at', { ascending: false });
  
  if (error) return res.status(500).json({ error: error.message });
  res.json({ documents: data });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));