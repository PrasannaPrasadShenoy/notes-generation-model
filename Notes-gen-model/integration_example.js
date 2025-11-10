/**
 * ILA Backend Integration Example
 * Shows how to integrate the Notes Generation API with your Node.js/Express backend
 */

const axios = require('axios');

// Configuration
const NOTES_API_URL = process.env.NOTES_API_URL || 'http://localhost:5000';

/**
 * Generate notes from transcript using the Notes Generation API
 */
async function generateNotes(transcript, options = {}) {
  try {
    const {
      type = 'enhanced',  // 'short', 'detailed', or 'enhanced'
      format = 'markdown',
      includeMetadata = true
    } = options;

    const response = await axios.post(`${NOTES_API_URL}/api/generate-notes`, {
      transcript: transcript,
      type: type,
      format: format,
      include_metadata: includeMetadata
    });

    if (response.data.success) {
      return {
        success: true,
        notes: response.data.notes,
        metadata: response.data.metadata
      };
    } else {
      throw new Error(response.data.error || 'Failed to generate notes');
    }
  } catch (error) {
    console.error('Error generating notes:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Generate enhanced notes v2 with all features
 */
async function generateEnhancedNotesV2(transcript, options = {}) {
  try {
    const {
      includeConcepts = true,
      includeTips = true,
      includeTopics = true,
      includeObjectives = true
    } = options;

    const response = await axios.post(`${NOTES_API_URL}/api/generate-enhanced-v2`, {
      transcript: transcript,
      include_concepts: includeConcepts,
      include_tips: includeTips,
      include_topics: includeTopics,
      include_objectives: includeObjectives
    });

    if (response.data.success) {
      return {
        success: true,
        notes: response.data.notes,
        baseNotes: response.data.base_notes,
        enhancements: response.data.enhancements,
        metadata: response.data.metadata
      };
    } else {
      throw new Error(response.data.error || 'Failed to generate enhanced notes');
    }
  } catch (error) {
    console.error('Error generating enhanced notes:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Generate notes for multiple videos in batch
 */
async function generateNotesBatch(transcripts, options = {}) {
  try {
    const {
      type = 'enhanced',
      format = 'markdown'
    } = options;

    const response = await axios.post(`${NOTES_API_URL}/api/generate-notes/batch`, {
      transcripts: transcripts.map(t => ({
        id: t.videoId || t.id,
        transcript: t.transcript
      })),
      type: type,
      format: format
    });

    if (response.data.success) {
      return {
        success: true,
        results: response.data.results,
        total: response.data.total,
        successful: response.data.successful
      };
    } else {
      throw new Error(response.data.error || 'Batch generation failed');
    }
  } catch (error) {
    console.error('Error in batch generation:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Analyze transcript without generating notes
 */
async function analyzeTranscript(transcript) {
  try {
    const response = await axios.post(`${NOTES_API_URL}/api/analyze-transcript`, {
      transcript: transcript
    });

    if (response.data.success) {
      return {
        success: true,
        analysis: response.data.analysis
      };
    } else {
      throw new Error(response.data.error || 'Analysis failed');
    }
  } catch (error) {
    console.error('Error analyzing transcript:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Example: Integration with your notesService.js
 */
async function generateNotesForVideo(videoId, transcript) {
  // First, analyze the transcript
  const analysis = await analyzeTranscript(transcript);
  
  if (!analysis.success) {
    throw new Error('Failed to analyze transcript');
  }

  // Check if transcript is suitable
  if (!analysis.analysis.is_suitable_for_notes) {
    throw new Error('Transcript too short for note generation');
  }

  // Generate enhanced notes
  const result = await generateEnhancedNotesV2(transcript, {
    includeConcepts: true,
    includeTips: true,
    includeTopics: true,
    includeObjectives: true
  });

  if (!result.success) {
    throw new Error(result.error);
  }

  // Return formatted result for your database
  return {
    videoId: videoId,
    notes: result.notes,
    baseNotes: result.baseNotes,
    keyConcepts: result.enhancements.key_concepts,
    studyTips: result.enhancements.study_tips,
    relatedTopics: result.enhancements.related_topics,
    learningObjectives: result.enhancements.learning_objectives,
    metadata: {
      readingTime: result.metadata.reading_time_minutes,
      wordCount: result.metadata.word_count,
      keyTopics: result.metadata.key_topics
    },
    generatedAt: new Date()
  };
}

// Example usage in Express route
/*
const express = require('express');
const router = express.Router();

router.post('/api/videos/:videoId/notes', async (req, res) => {
  try {
    const { videoId } = req.params;
    const { transcript } = req.body;

    if (!transcript) {
      return res.status(400).json({ error: 'Transcript is required' });
    }

    const notes = await generateNotesForVideo(videoId, transcript);
    
    // Save to your database
    // await NotesModel.create(notes);

    res.json({
      success: true,
      notes: notes
    });
  } catch (error) {
    console.error('Error generating notes:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});
*/

module.exports = {
  generateNotes,
  generateEnhancedNotesV2,
  generateNotesBatch,
  analyzeTranscript,
  generateNotesForVideo
};

