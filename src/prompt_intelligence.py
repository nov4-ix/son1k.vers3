"""
Intelligent Prompt Enhancement Engine for Son1kVers3
Converts vague musical ideas into precise, actionable instructions for AI generation
"""

import re
import logging
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

class PromptComplexity(Enum):
    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class MusicalConcept:
    """Represents a musical concept with associated keywords and parameters"""
    category: str
    keywords: List[str]
    parameters: Dict[str, Any]
    confidence_boost: float = 1.0
    exclusions: List[str] = None

@dataclass
class EnhancementResult:
    """Result of prompt enhancement"""
    original_prompt: str
    enhanced_prompt: str
    detected_elements: Dict[str, Any]
    confidence_score: float
    suggestions: List[str]
    complexity: PromptComplexity
    metadata: Dict[str, Any]

class PromptIntelligenceEngine:
    """Advanced prompt enhancement and analysis engine"""
    
    def __init__(self):
        self.musical_concepts = self._initialize_musical_concepts()
        self.style_templates = self._initialize_style_templates()
        self.emotion_mappings = self._initialize_emotion_mappings()
        self.instrument_families = self._initialize_instrument_families()
        self.genre_characteristics = self._initialize_genre_characteristics()
        self.cultural_contexts = self._initialize_cultural_contexts()
        
        # Enhancement patterns
        self.enhancement_patterns = self._initialize_enhancement_patterns()
        
        logger.info("Prompt Intelligence Engine initialized with comprehensive musical knowledge")
    
    def _initialize_musical_concepts(self) -> Dict[str, MusicalConcept]:
        """Initialize comprehensive musical concept database"""
        concepts = {}
        
        # Genres
        concepts['rock'] = MusicalConcept(
            category='genre',
            keywords=['rock', 'guitar', 'drums', 'electric', 'distortion', 'power chords'],
            parameters={
                'tempo_range': (90, 160),
                'key_preferences': ['E', 'A', 'D', 'G'],
                'scale_types': ['minor', 'blues', 'dorian'],
                'dynamics': 'loud',
                'production_style': 'punchy'
            }
        )
        
        concepts['pop'] = MusicalConcept(
            category='genre',
            keywords=['pop', 'catchy', 'mainstream', 'hooks', 'commercial', 'radio'],
            parameters={
                'tempo_range': (100, 140),
                'key_preferences': ['C', 'G', 'F', 'Am'],
                'scale_types': ['major', 'minor'],
                'structure': 'verse-chorus',
                'production_style': 'polished'
            }
        )
        
        concepts['jazz'] = MusicalConcept(
            category='genre',
            keywords=['jazz', 'swing', 'bebop', 'fusion', 'improvisation', 'complex harmony'],
            parameters={
                'tempo_range': (60, 200),
                'key_preferences': ['Bb', 'F', 'Eb', 'Dm'],
                'scale_types': ['major', 'minor', 'dorian', 'mixolydian'],
                'chord_complexity': 'extended',
                'rhythm': 'swing'
            }
        )
        
        concepts['electronic'] = MusicalConcept(
            category='genre',
            keywords=['electronic', 'synth', 'edm', 'techno', 'house', 'digital', 'programmed'],
            parameters={
                'tempo_range': (120, 150),
                'key_preferences': ['Am', 'Dm', 'Em', 'Gm'],
                'scale_types': ['minor', 'dorian'],
                'production_style': 'digital',
                'rhythm': 'quantized'
            }
        )
        
        concepts['classical'] = MusicalConcept(
            category='genre',
            keywords=['classical', 'orchestral', 'symphony', 'baroque', 'romantic', 'chamber'],
            parameters={
                'tempo_range': (40, 180),
                'key_preferences': ['C', 'G', 'D', 'F'],
                'scale_types': ['major', 'minor'],
                'instrumentation': 'orchestral',
                'dynamics': 'varied'
            }
        )
        
        # Emotions
        concepts['happy'] = MusicalConcept(
            category='emotion',
            keywords=['happy', 'joyful', 'uplifting', 'cheerful', 'bright', 'optimistic'],
            parameters={
                'tempo_boost': 1.1,
                'scale_preference': 'major',
                'rhythm': 'bouncy',
                'dynamics': 'bright'
            }
        )
        
        concepts['sad'] = MusicalConcept(
            category='emotion',
            keywords=['sad', 'melancholic', 'sorrowful', 'emotional', 'tearful', 'blue'],
            parameters={
                'tempo_boost': 0.8,
                'scale_preference': 'minor',
                'rhythm': 'slow',
                'dynamics': 'soft'
            }
        )
        
        concepts['energetic'] = MusicalConcept(
            category='emotion',
            keywords=['energetic', 'powerful', 'driving', 'intense', 'aggressive', 'fierce'],
            parameters={
                'tempo_boost': 1.3,
                'dynamics': 'loud',
                'rhythm': 'driving',
                'intensity': 'high'
            }
        )
        
        concepts['calm'] = MusicalConcept(
            category='emotion',
            keywords=['calm', 'peaceful', 'serene', 'gentle', 'soft', 'relaxing'],
            parameters={
                'tempo_boost': 0.7,
                'dynamics': 'soft',
                'rhythm': 'flowing',
                'texture': 'smooth'
            }
        )
        
        # Instruments
        concepts['piano'] = MusicalConcept(
            category='instrument',
            keywords=['piano', 'keyboard', 'keys', 'ivory', 'grand piano', 'upright'],
            parameters={
                'range': (27.5, 4186),
                'polyphony': 'unlimited',
                'expression': 'dynamic',
                'timbre': 'percussive'
            }
        )
        
        concepts['guitar'] = MusicalConcept(
            category='instrument',
            keywords=['guitar', 'acoustic guitar', 'electric guitar', 'strings', 'fingerpicking'],
            parameters={
                'range': (82, 1175),
                'polyphony': 6,
                'expression': 'varied',
                'techniques': ['strumming', 'picking', 'bending']
            }
        )
        
        concepts['strings'] = MusicalConcept(
            category='instrument',
            keywords=['strings', 'violin', 'viola', 'cello', 'orchestra', 'bowing'],
            parameters={
                'range': (65, 2093),
                'expression': 'legato',
                'dynamics': 'smooth',
                'timbre': 'warm'
            }
        )
        
        concepts['brass'] = MusicalConcept(
            category='instrument',
            keywords=['brass', 'trumpet', 'trombone', 'horn', 'fanfare', 'bold'],
            parameters={
                'range': (58, 1760),
                'expression': 'bold',
                'dynamics': 'loud',
                'timbre': 'bright'
            }
        )
        
        return concepts
    
    def _initialize_style_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize style-specific enhancement templates"""
        return {
            'rock_ballad': {
                'template': "A {emotion} rock ballad with {instruments}, featuring {tempo} tempo and {key} key",
                'required_elements': ['emotion', 'tempo', 'key'],
                'default_instruments': 'electric guitar and piano',
                'tempo_range': (60, 90)
            },
            'upbeat_pop': {
                'template': "An upbeat pop song with {instruments}, {tempo} BPM, infectious melody in {key}",
                'required_elements': ['tempo', 'key'],
                'default_instruments': 'synthesizers and drums',
                'tempo_range': (110, 140)
            },
            'ambient_electronic': {
                'template': "Ambient electronic soundscape with {instruments}, slow evolving textures, {key} tonality",
                'required_elements': ['key'],
                'default_instruments': 'synthesizers and pads',
                'tempo_range': (60, 90)
            },
            'latin_rhythm': {
                'template': "Latin music with {rhythm} rhythm, {instruments}, {tempo} BPM in {key}",
                'required_elements': ['rhythm', 'tempo', 'key'],
                'default_instruments': 'guitar, piano, and percussion',
                'tempo_range': (100, 140)
            },
            'jazz_standard': {
                'template': "Jazz standard with {instruments}, swing feel, {key} with extended harmonies",
                'required_elements': ['key'],
                'default_instruments': 'piano, bass, and drums',
                'tempo_range': (80, 160)
            }
        }
    
    def _initialize_emotion_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize emotion to musical parameter mappings"""
        return {
            'love': {
                'tempo_modifier': 0.9,
                'key_preference': 'major',
                'suggested_instruments': ['piano', 'strings', 'soft vocals'],
                'dynamics': 'intimate',
                'suggested_words': ['romantic', 'tender', 'warm', 'gentle']
            },
            'anger': {
                'tempo_modifier': 1.3,
                'key_preference': 'minor',
                'suggested_instruments': ['electric guitar', 'drums', 'distortion'],
                'dynamics': 'aggressive',
                'suggested_words': ['fierce', 'intense', 'driving', 'powerful']
            },
            'nostalgia': {
                'tempo_modifier': 0.8,
                'key_preference': 'minor',
                'suggested_instruments': ['acoustic guitar', 'piano', 'strings'],
                'dynamics': 'wistful',
                'suggested_words': ['memories', 'vintage', 'warm', 'golden']
            },
            'triumph': {
                'tempo_modifier': 1.2,
                'key_preference': 'major',
                'suggested_instruments': ['brass', 'orchestra', 'choir'],
                'dynamics': 'bold',
                'suggested_words': ['victorious', 'soaring', 'majestic', 'epic']
            }
        }
    
    def _initialize_instrument_families(self) -> Dict[str, List[str]]:
        """Initialize instrument family classifications"""
        return {
            'strings': ['violin', 'viola', 'cello', 'double bass', 'guitar', 'harp'],
            'woodwinds': ['flute', 'clarinet', 'oboe', 'bassoon', 'saxophone'],
            'brass': ['trumpet', 'trombone', 'french horn', 'tuba'],
            'percussion': ['drums', 'timpani', 'cymbals', 'xylophone', 'marimba'],
            'keyboard': ['piano', 'organ', 'harpsichord', 'synthesizer'],
            'vocals': ['soprano', 'alto', 'tenor', 'bass', 'choir'],
            'electronic': ['synthesizer', 'sampler', 'drum machine', 'sequencer']
        }
    
    def _initialize_genre_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize detailed genre characteristics"""
        return {
            'blues': {
                'scales': ['blues', 'minor pentatonic'],
                'chord_progressions': ['12-bar blues', 'I-IV-V'],
                'typical_instruments': ['guitar', 'harmonica', 'piano'],
                'tempo_range': (60, 120),
                'characteristics': ['blue notes', 'call and response', 'bent notes']
            },
            'reggae': {
                'rhythm_patterns': ['off-beat emphasis', 'skank'],
                'typical_instruments': ['guitar', 'bass', 'drums'],
                'tempo_range': (90, 120),
                'characteristics': ['syncopation', 'off-beat chords', 'relaxed feel']
            },
            'country': {
                'typical_instruments': ['acoustic guitar', 'fiddle', 'banjo', 'steel guitar'],
                'scales': ['major', 'mixolydian'],
                'tempo_range': (80, 140),
                'characteristics': ['storytelling', 'twang', 'simple harmonies']
            },
            'funk': {
                'rhythm_patterns': ['groove', 'syncopation'],
                'typical_instruments': ['bass', 'drums', 'electric guitar'],
                'tempo_range': (90, 130),
                'characteristics': ['rhythmic emphasis', 'tight groove', 'percussive']
            }
        }
    
    def _initialize_cultural_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural and regional musical contexts"""
        return {
            'latin': {
                'subgenres': ['salsa', 'bossa nova', 'tango', 'reggaeton', 'mariachi'],
                'instruments': ['guitar', 'piano', 'brass', 'percussion'],
                'rhythms': ['clave', 'montuno', 'bossa nova'],
                'languages': ['spanish', 'portuguese'],
                'characteristics': ['syncopation', 'polyrhythm', 'passion']
            },
            'asian': {
                'subgenres': ['j-pop', 'k-pop', 'traditional', 'gamelan'],
                'instruments': ['shamisen', 'koto', 'erhu', 'gamelan'],
                'scales': ['pentatonic', 'whole tone'],
                'characteristics': ['minimalism', 'precision', 'harmony']
            },
            'african': {
                'subgenres': ['afrobeat', 'highlife', 'soukous'],
                'instruments': ['drums', 'kalimba', 'kora'],
                'rhythms': ['polyrhythm', 'cross-rhythm'],
                'characteristics': ['complex rhythms', 'call and response', 'community']
            },
            'middle_eastern': {
                'instruments': ['oud', 'qanun', 'ney', 'darbuka'],
                'scales': ['maqam', 'harmonic minor'],
                'characteristics': ['microtones', 'ornamentation', 'modal']
            }
        }
    
    def _initialize_enhancement_patterns(self) -> List[Dict[str, Any]]:
        """Initialize prompt enhancement patterns"""
        return [
            {
                'pattern': r'\b(slow|fast|quick|rapid)\b',
                'category': 'tempo',
                'enhancement': lambda m: self._enhance_tempo_description(m.group(1))
            },
            {
                'pattern': r'\b(guitar|piano|drums|violin)\b',
                'category': 'instruments',
                'enhancement': lambda m: self._enhance_instrument_description(m.group(1))
            },
            {
                'pattern': r'\b(happy|sad|angry|calm)\b',
                'category': 'emotion',
                'enhancement': lambda m: self._enhance_emotion_description(m.group(1))
            },
            {
                'pattern': r'\b(major|minor)\b',
                'category': 'harmony',
                'enhancement': lambda m: self._enhance_harmony_description(m.group(1))
            }
        ]
    
    def enhance_prompt(self, 
                      prompt: str, 
                      target_complexity: PromptComplexity = PromptComplexity.DETAILED,
                      style_hint: Optional[str] = None,
                      cultural_context: Optional[str] = None) -> EnhancementResult:
        """
        Enhance a musical prompt with intelligent analysis and expansion
        
        Args:
            prompt: Original prompt
            target_complexity: Desired complexity level
            style_hint: Optional style guidance
            cultural_context: Optional cultural context
        
        Returns:
            EnhancementResult with enhanced prompt and analysis
        """
        
        logger.info(f"Enhancing prompt: '{prompt[:50]}...'")
        
        # 1. Analyze the original prompt
        analysis = self._analyze_prompt(prompt)
        
        # 2. Detect missing critical elements
        missing_elements = self._identify_missing_elements(analysis, target_complexity)
        
        # 3. Generate enhancements
        enhancements = self._generate_enhancements(
            prompt, analysis, missing_elements, style_hint, cultural_context
        )
        
        # 4. Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(
            prompt, enhancements, target_complexity
        )
        
        # 5. Calculate confidence score
        confidence = self._calculate_confidence_score(analysis, enhancements)
        
        # 6. Generate suggestions
        suggestions = self._generate_suggestions(analysis, missing_elements)
        
        result = EnhancementResult(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            detected_elements=analysis,
            confidence_score=confidence,
            suggestions=suggestions,
            complexity=target_complexity,
            metadata={
                'enhancements_applied': len(enhancements),
                'missing_elements': missing_elements,
                'style_hint': style_hint,
                'cultural_context': cultural_context
            }
        )
        
        logger.info(f"Enhancement complete. Confidence: {confidence:.2f}")
        
        return result
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt to extract musical elements"""
        prompt_lower = prompt.lower()
        analysis = {
            'genres': [],
            'instruments': [],
            'emotions': [],
            'tempo_indicators': [],
            'key_indicators': [],
            'structure_elements': [],
            'production_elements': [],
            'cultural_elements': [],
            'technical_elements': []
        }
        
        # Analyze each concept category
        for concept_name, concept in self.musical_concepts.items():
            for keyword in concept.keywords:
                if keyword in prompt_lower:
                    category_key = concept.category + 's'  # Pluralize
                    if category_key in analysis:
                        analysis[category_key].append({
                            'concept': concept_name,
                            'keyword': keyword,
                            'parameters': concept.parameters,
                            'confidence': self._calculate_keyword_confidence(keyword, prompt_lower)
                        })
        
        # Detect specific musical elements
        analysis['tempo_indicators'] = self._detect_tempo_indicators(prompt_lower)
        analysis['key_indicators'] = self._detect_key_indicators(prompt_lower)
        analysis['time_signatures'] = self._detect_time_signatures(prompt_lower)
        analysis['chord_progressions'] = self._detect_chord_progressions(prompt_lower)
        
        # Analyze complexity and completeness
        analysis['complexity_score'] = self._assess_prompt_complexity(prompt)
        analysis['completeness_score'] = self._assess_prompt_completeness(analysis)
        
        return analysis
    
    def _detect_tempo_indicators(self, prompt: str) -> List[Dict[str, Any]]:
        """Detect tempo-related information in prompt"""
        tempo_patterns = {
            r'(\d+)\s*bpm': lambda m: {'type': 'explicit', 'value': int(m.group(1))},
            r'\b(slow|slowly)\b': lambda m: {'type': 'qualitative', 'range': (60, 80)},
            r'\b(medium|moderate)\b': lambda m: {'type': 'qualitative', 'range': (90, 110)},
            r'\b(fast|quickly|rapid)\b': lambda m: {'type': 'qualitative', 'range': (120, 150)},
            r'\b(ballad)\b': lambda m: {'type': 'style', 'range': (60, 80)},
            r'\b(dance|dancing)\b': lambda m: {'type': 'style', 'range': (110, 130)}
        }
        
        indicators = []
        for pattern, extractor in tempo_patterns.items():
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                indicators.append(extractor(match))
        
        return indicators
    
    def _detect_key_indicators(self, prompt: str) -> List[Dict[str, Any]]:
        """Detect key and scale information"""
        key_patterns = {
            r'\b([ABCDEFG][#b]?)\s+(major|minor)\b': lambda m: {
                'root': m.group(1), 'scale': m.group(2), 'confidence': 0.9
            },
            r'\bin\s+([ABCDEFG][#b]?)\b': lambda m: {
                'root': m.group(1), 'scale': 'major', 'confidence': 0.6
            },
            r'\b(blues|pentatonic|dorian|mixolydian)\b': lambda m: {
                'scale': m.group(1), 'confidence': 0.7
            }
        }
        
        indicators = []
        for pattern, extractor in key_patterns.items():
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                indicators.append(extractor(match))
        
        return indicators
    
    def _detect_time_signatures(self, prompt: str) -> List[str]:
        """Detect time signature indicators"""
        time_patterns = [
            r'\b(\d+/\d+)\s*time\b',
            r'\b(waltz)\b',  # Implies 3/4
            r'\b(march)\b',  # Implies 4/4
            r'\b(swing)\b'   # Implies compound time
        ]
        
        signatures = []
        for pattern in time_patterns:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            signatures.extend([m.group(1) for m in matches])
        
        return signatures
    
    def _detect_chord_progressions(self, prompt: str) -> List[str]:
        """Detect chord progression mentions"""
        progression_patterns = [
            r'\b(I-V-vi-IV)\b',
            r'\b(ii-V-I)\b',
            r'\b(12-bar blues)\b',
            r'\b(circle of fifths)\b'
        ]
        
        progressions = []
        for pattern in progression_patterns:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            progressions.extend([m.group(1) for m in matches])
        
        return progressions
    
    def _calculate_keyword_confidence(self, keyword: str, prompt: str) -> float:
        """Calculate confidence score for keyword detection"""
        base_confidence = 0.7
        
        # Boost confidence for exact matches
        if keyword in prompt:
            base_confidence += 0.2
        
        # Context boosting
        context_words = prompt.split()
        keyword_index = -1
        
        for i, word in enumerate(context_words):
            if keyword in word.lower():
                keyword_index = i
                break
        
        if keyword_index >= 0:
            # Check surrounding context
            context_start = max(0, keyword_index - 2)
            context_end = min(len(context_words), keyword_index + 3)
            context = ' '.join(context_words[context_start:context_end]).lower()
            
            # Musical context words boost confidence
            musical_context = ['music', 'song', 'track', 'melody', 'rhythm', 'style', 'genre']
            for ctx_word in musical_context:
                if ctx_word in context:
                    base_confidence += 0.1
                    break
        
        return min(base_confidence, 1.0)
    
    def _assess_prompt_complexity(self, prompt: str) -> float:
        """Assess the complexity of the original prompt"""
        complexity_indicators = {
            'technical_terms': ['BPM', 'key', 'scale', 'chord', 'progression', 'time signature'],
            'specific_instruments': ['guitar', 'piano', 'violin', 'drums', 'synthesizer'],
            'production_terms': ['reverb', 'compression', 'EQ', 'distortion', 'mixing'],
            'advanced_concepts': ['polyrhythm', 'modulation', 'counterpoint', 'harmony']
        }
        
        word_count = len(prompt.split())
        score = min(word_count / 20, 0.4)  # Base score from length
        
        prompt_lower = prompt.lower()
        for category, terms in complexity_indicators.items():
            for term in terms:
                if term.lower() in prompt_lower:
                    if category == 'advanced_concepts':
                        score += 0.2
                    elif category == 'production_terms':
                        score += 0.15
                    elif category == 'technical_terms':
                        score += 0.1
                    else:
                        score += 0.05
        
        return min(score, 1.0)
    
    def _assess_prompt_completeness(self, analysis: Dict[str, Any]) -> float:
        """Assess how complete the prompt is"""
        essential_elements = ['genres', 'instruments', 'emotions', 'tempo_indicators']
        nice_to_have = ['key_indicators', 'structure_elements', 'production_elements']
        
        score = 0.0
        
        # Check essential elements
        for element in essential_elements:
            if analysis.get(element) and len(analysis[element]) > 0:
                score += 0.2
        
        # Check nice-to-have elements
        for element in nice_to_have:
            if analysis.get(element) and len(analysis[element]) > 0:
                score += 0.1
        
        return min(score, 1.0)
    
    def _identify_missing_elements(self, 
                                  analysis: Dict[str, Any], 
                                  target_complexity: PromptComplexity) -> List[str]:
        """Identify missing elements based on target complexity"""
        missing = []
        
        # Essential elements for all complexity levels
        if not analysis.get('genres') or len(analysis['genres']) == 0:
            missing.append('genre')
        
        if not analysis.get('tempo_indicators') or len(analysis['tempo_indicators']) == 0:
            missing.append('tempo')
        
        # Additional elements for higher complexity
        if target_complexity in [PromptComplexity.DETAILED, PromptComplexity.COMPREHENSIVE]:
            if not analysis.get('instruments') or len(analysis['instruments']) == 0:
                missing.append('instruments')
            
            if not analysis.get('key_indicators') or len(analysis['key_indicators']) == 0:
                missing.append('key')
            
            if not analysis.get('emotions') or len(analysis['emotions']) == 0:
                missing.append('emotion')
        
        # Expert elements for comprehensive complexity
        if target_complexity == PromptComplexity.COMPREHENSIVE:
            if not analysis.get('structure_elements') or len(analysis['structure_elements']) == 0:
                missing.append('structure')
            
            if not analysis.get('production_elements') or len(analysis['production_elements']) == 0:
                missing.append('production')
        
        return missing
    
    def _generate_enhancements(self, 
                              prompt: str,
                              analysis: Dict[str, Any],
                              missing_elements: List[str],
                              style_hint: Optional[str],
                              cultural_context: Optional[str]) -> Dict[str, Any]:
        """Generate specific enhancements for the prompt"""
        enhancements = {}
        
        # Handle missing genre
        if 'genre' in missing_elements:
            if style_hint:
                enhancements['genre'] = style_hint
            else:
                # Infer genre from other elements
                enhancements['genre'] = self._infer_genre(analysis)
        
        # Handle missing tempo
        if 'tempo' in missing_elements:
            enhancements['tempo'] = self._suggest_tempo(analysis, enhancements.get('genre'))
        
        # Handle missing instruments
        if 'instruments' in missing_elements:
            enhancements['instruments'] = self._suggest_instruments(analysis, enhancements.get('genre'))
        
        # Handle missing key
        if 'key' in missing_elements:
            enhancements['key'] = self._suggest_key(analysis, enhancements.get('genre'))
        
        # Handle missing emotion
        if 'emotion' in missing_elements:
            enhancements['emotion'] = self._infer_emotion(prompt, analysis)
        
        # Handle missing structure
        if 'structure' in missing_elements:
            enhancements['structure'] = self._suggest_structure(enhancements.get('genre'))
        
        # Handle missing production
        if 'production' in missing_elements:
            enhancements['production'] = self._suggest_production(enhancements.get('genre'))
        
        # Cultural context enhancements
        if cultural_context and cultural_context in self.cultural_contexts:
            enhancements['cultural'] = self._apply_cultural_context(cultural_context)
        
        return enhancements
    
    def _infer_genre(self, analysis: Dict[str, Any]) -> str:
        """Infer genre from available analysis"""
        # Check instruments for genre clues
        instruments = analysis.get('instruments', [])
        
        for instr in instruments:
            concept = instr.get('concept', '')
            if concept == 'guitar' and any('electric' in k.get('keyword', '') for k in instruments):
                return 'rock'
            elif concept == 'piano':
                return 'classical'
        
        # Check emotions for genre clues
        emotions = analysis.get('emotions', [])
        for emotion in emotions:
            concept = emotion.get('concept', '')
            if concept in ['energetic', 'angry']:
                return 'rock'
            elif concept == 'calm':
                return 'ambient'
        
        # Default to pop
        return 'pop'
    
    def _suggest_tempo(self, analysis: Dict[str, Any], genre: Optional[str]) -> str:
        """Suggest appropriate tempo"""
        if genre and genre in self.musical_concepts:
            params = self.musical_concepts[genre].parameters
            tempo_range = params.get('tempo_range', (100, 120))
            suggested_tempo = (tempo_range[0] + tempo_range[1]) // 2
            return f"{suggested_tempo} BPM"
        
        # Check emotions for tempo clues
        emotions = analysis.get('emotions', [])
        for emotion in emotions:
            concept = emotion.get('concept', '')
            if concept in ['energetic', 'happy']:
                return "120 BPM"
            elif concept in ['sad', 'calm']:
                return "80 BPM"
        
        return "110 BPM"  # Default moderate tempo
    
    def _suggest_instruments(self, analysis: Dict[str, Any], genre: Optional[str]) -> str:
        """Suggest appropriate instruments"""
        if genre in ['rock', 'pop']:
            return "electric guitar, bass, and drums"
        elif genre == 'jazz':
            return "piano, upright bass, and drums"
        elif genre == 'classical':
            return "strings and piano"
        elif genre == 'electronic':
            return "synthesizers and programmed drums"
        else:
            return "piano and gentle percussion"
    
    def _suggest_key(self, analysis: Dict[str, Any], genre: Optional[str]) -> str:
        """Suggest appropriate musical key"""
        if genre and genre in self.musical_concepts:
            params = self.musical_concepts[genre].parameters
            key_prefs = params.get('key_preferences', ['C'])
            scale_types = params.get('scale_types', ['major'])
            
            key = random.choice(key_prefs)
            scale = random.choice(scale_types)
            return f"{key} {scale}"
        
        # Check emotions for key suggestions
        emotions = analysis.get('emotions', [])
        for emotion in emotions:
            concept = emotion.get('concept', '')
            if concept in ['happy', 'energetic']:
                return "C major"
            elif concept in ['sad', 'melancholic']:
                return "A minor"
        
        return "C major"  # Default
    
    def _infer_emotion(self, prompt: str, analysis: Dict[str, Any]) -> str:
        """Infer emotion from prompt context"""
        emotion_keywords = {
            'love': ['love', 'heart', 'romance', 'tender'],
            'joy': ['celebration', 'party', 'dancing', 'bright'],
            'melancholy': ['rain', 'memory', 'lost', 'goodbye'],
            'power': ['strong', 'force', 'victory', 'triumph']
        }
        
        prompt_lower = prompt.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return emotion
        
        return "uplifting"  # Default positive emotion
    
    def _suggest_structure(self, genre: Optional[str]) -> str:
        """Suggest musical structure"""
        structures = {
            'pop': "verse-chorus-verse-chorus-bridge-chorus",
            'rock': "intro-verse-chorus-verse-chorus-solo-chorus",
            'jazz': "head-improvisation-head",
            'classical': "exposition-development-recapitulation",
            'electronic': "intro-buildup-drop-breakdown-drop"
        }
        
        return structures.get(genre, "verse-chorus structure")
    
    def _suggest_production(self, genre: Optional[str]) -> str:
        """Suggest production characteristics"""
        production = {
            'pop': "polished, bright mix with clear vocals",
            'rock': "punchy drums, distorted guitars, powerful mix",
            'jazz': "natural reverb, warm tone, dynamic range",
            'classical': "spacious acoustics, natural dynamics",
            'electronic': "tight, compressed, with digital effects"
        }
        
        return production.get(genre, "clean, balanced mix")
    
    def _apply_cultural_context(self, context: str) -> Dict[str, Any]:
        """Apply cultural context enhancements"""
        return self.cultural_contexts.get(context, {})
    
    def _build_enhanced_prompt(self, 
                              original: str,
                              enhancements: Dict[str, Any],
                              complexity: PromptComplexity) -> str:
        """Build the enhanced prompt"""
        
        # Start with original prompt
        enhanced = original.strip()
        
        # Add enhancements based on complexity level
        additions = []
        
        if 'genre' in enhancements:
            additions.append(f"in {enhancements['genre']} style")
        
        if 'tempo' in enhancements:
            additions.append(f"at {enhancements['tempo']}")
        
        if complexity in [PromptComplexity.DETAILED, PromptComplexity.COMPREHENSIVE]:
            if 'instruments' in enhancements:
                additions.append(f"featuring {enhancements['instruments']}")
            
            if 'key' in enhancements:
                additions.append(f"in {enhancements['key']}")
            
            if 'emotion' in enhancements:
                additions.append(f"with {enhancements['emotion']} feeling")
        
        if complexity == PromptComplexity.COMPREHENSIVE:
            if 'structure' in enhancements:
                additions.append(f"using {enhancements['structure']}")
            
            if 'production' in enhancements:
                additions.append(f"with {enhancements['production']}")
        
        # Combine original with additions
        if additions:
            enhanced += ", " + ", ".join(additions)
        
        return enhanced
    
    def _calculate_confidence_score(self, 
                                   analysis: Dict[str, Any], 
                                   enhancements: Dict[str, Any]) -> float:
        """Calculate confidence score for the enhancement"""
        base_score = analysis.get('completeness_score', 0.5)
        enhancement_boost = len(enhancements) * 0.1
        complexity_boost = analysis.get('complexity_score', 0.0) * 0.2
        
        return min(base_score + enhancement_boost + complexity_boost, 1.0)
    
    def _generate_suggestions(self, 
                            analysis: Dict[str, Any], 
                            missing_elements: List[str]) -> List[str]:
        """Generate helpful suggestions for further improvement"""
        suggestions = []
        
        if 'instruments' in missing_elements:
            suggestions.append("Consider specifying instruments for more precise generation")
        
        if 'key' in missing_elements:
            suggestions.append("Adding a musical key can improve harmonic coherence")
        
        if 'structure' in missing_elements:
            suggestions.append("Describing song structure helps with arrangement")
        
        if analysis.get('complexity_score', 0) < 0.3:
            suggestions.append("More detailed descriptions typically yield better results")
        
        if not analysis.get('cultural_elements'):
            suggestions.append("Cultural or regional style references can add authenticity")
        
        return suggestions
    
    def _enhance_tempo_description(self, tempo_word: str) -> str:
        """Enhance tempo descriptions"""
        enhancements = {
            'slow': 'slow and contemplative (70 BPM)',
            'fast': 'fast and energetic (130 BPM)',
            'quick': 'quick and lively (120 BPM)',
            'rapid': 'rapid and intense (140 BPM)'
        }
        return enhancements.get(tempo_word.lower(), tempo_word)
    
    def _enhance_instrument_description(self, instrument: str) -> str:
        """Enhance instrument descriptions"""
        enhancements = {
            'guitar': 'electric guitar with warm distortion',
            'piano': 'grand piano with rich tone',
            'drums': 'dynamic drum kit with crisp snare',
            'violin': 'expressive violin with vibrato'
        }
        return enhancements.get(instrument.lower(), instrument)
    
    def _enhance_emotion_description(self, emotion: str) -> str:
        """Enhance emotion descriptions"""
        enhancements = {
            'happy': 'joyful and uplifting',
            'sad': 'melancholic and emotional',
            'angry': 'intense and powerful',
            'calm': 'peaceful and serene'
        }
        return enhancements.get(emotion.lower(), emotion)
    
    def _enhance_harmony_description(self, harmony: str) -> str:
        """Enhance harmony descriptions"""
        enhancements = {
            'major': 'bright major tonality',
            'minor': 'expressive minor key'
        }
        return enhancements.get(harmony.lower(), harmony)

# Global engine instance
_prompt_engine: Optional[PromptIntelligenceEngine] = None

def get_prompt_engine() -> PromptIntelligenceEngine:
    """Get the global prompt intelligence engine"""
    global _prompt_engine
    if _prompt_engine is None:
        _prompt_engine = PromptIntelligenceEngine()
    return _prompt_engine

def enhance_prompt_simple(prompt: str, style: Optional[str] = None) -> str:
    """Simple prompt enhancement function"""
    engine = get_prompt_engine()
    result = engine.enhance_prompt(prompt, style_hint=style)
    return result.enhanced_prompt

if __name__ == "__main__":
    # Test the prompt enhancement engine
    engine = PromptIntelligenceEngine()
    
    test_prompts = [
        "a sad song",
        "upbeat music with guitar",
        "classical piece in C major",
        "electronic dance track 128 BPM"
    ]
    
    for prompt in test_prompts:
        result = engine.enhance_prompt(prompt, PromptComplexity.DETAILED)
        print(f"Original: {prompt}")
        print(f"Enhanced: {result.enhanced_prompt}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print("---")