"""
Advanced Lyrics AI System
Provides narrative coherence, rhyme scheme intelligence, and sophisticated lyric generation
"""
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
import json
import random
from pathlib import Path
import nltk
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            pass

    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        try:
            nltk.download('cmudict')
        except:
            pass
except ImportError:
    NLTK_AVAILABLE = False

class LyricsTheme(Enum):
    """Common lyrical themes"""
    LOVE = "love"
    HEARTBREAK = "heartbreak"
    FREEDOM = "freedom"
    HOPE = "hope"
    REBELLION = "rebellion"
    NOSTALGIA = "nostalgia"
    PARTY = "party"
    SOCIAL_ISSUES = "social_issues"
    PERSONAL_GROWTH = "personal_growth"
    SPIRITUALITY = "spirituality"
    NATURE = "nature"
    FRIENDSHIP = "friendship"
    FAMILY = "family"
    SUCCESS = "success"
    STRUGGLE = "struggle"

class RhymeScheme(Enum):
    """Common rhyme schemes"""
    AABB = "aabb"
    ABAB = "abab"
    ABCB = "abcb"
    AAAA = "aaaa"
    ABBA = "abba"
    AABA = "aaba"
    FREE = "free"

class LyricStyle(Enum):
    """Lyrical styles"""
    NARRATIVE = "narrative"
    CONVERSATIONAL = "conversational"
    METAPHORICAL = "metaphorical"
    ABSTRACT = "abstract"
    DIRECT = "direct"
    POETIC = "poetic"
    RAP = "rap"
    FOLK = "folk"

class EmotionalArc(Enum):
    """Emotional progression patterns"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    U_SHAPED = "u_shaped"
    INVERTED_U = "inverted_u"
    COMPLEX = "complex"

@dataclass
class RhymePattern:
    """Represents a rhyme pattern"""
    scheme: RhymeScheme
    pattern: List[str]
    syllable_counts: List[int]
    stress_patterns: List[str]

@dataclass
class LyricLine:
    """Represents a single line of lyrics"""
    text: str
    rhyme_sound: str
    syllable_count: int
    stress_pattern: str
    emotional_weight: float
    theme_relevance: float
    phonemes: List[str]

@dataclass
class LyricVerse:
    """Represents a verse structure"""
    lines: List[LyricLine]
    verse_type: str  # verse, chorus, bridge, pre_chorus
    rhyme_scheme: RhymeScheme
    emotional_intensity: float
    theme_focus: LyricsTheme
    narrative_function: str

@dataclass
class NarrativeStructure:
    """Represents the narrative structure of lyrics"""
    exposition: str
    rising_action: str
    climax: str
    falling_action: str
    resolution: str
    emotional_arc: EmotionalArc
    character_development: Dict[str, Any]
    plot_points: List[str]

@dataclass
class LyricalCoherence:
    """Measures of lyrical coherence"""
    thematic_consistency: float
    narrative_flow: float
    emotional_progression: float
    linguistic_coherence: float
    structural_integrity: float
    overall_score: float

@dataclass
class LyricsGenerationRequest:
    """Request for lyrics generation"""
    theme: LyricsTheme
    style: LyricStyle
    rhyme_scheme: RhymeScheme
    verse_count: int = 2
    chorus_count: int = 1
    bridge_count: int = 1
    emotional_arc: EmotionalArc = EmotionalArc.U_SHAPED
    target_length: int = 100  # words
    complexity_level: float = 0.7  # 0-1
    language_register: str = "casual"  # casual, formal, poetic
    cultural_context: str = "contemporary"

class PhoneticAnalyzer:
    """Analyzes phonetic properties for rhyming"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                from nltk.corpus import cmudict
                self.phoneme_dict = cmudict.dict()
            except Exception as e:
                logger.warning(f"CMU dict not available: {e}")
                self.phoneme_dict = {}
        else:
            logger.warning("NLTK not available - using simplified phonetic analysis")
            self.phoneme_dict = {}
    
    def get_phonemes(self, word: str) -> List[str]:
        """Get phonemes for a word"""
        try:
            word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
            if word_clean in self.phoneme_dict:
                return self.phoneme_dict[word_clean][0]  # First pronunciation
            else:
                return self._approximate_phonemes(word_clean)
        except:
            return []
    
    def _approximate_phonemes(self, word: str) -> List[str]:
        """Approximate phonemes for unknown words"""
        # Simplified phoneme approximation
        vowels = set('aeiou')
        phonemes = []
        
        for i, char in enumerate(word):
            if char in vowels:
                phonemes.append(char.upper())
            else:
                phonemes.append(char.upper())
        
        return phonemes
    
    def get_rhyme_sound(self, word: str) -> str:
        """Extract rhyming sound from word"""
        try:
            phonemes = self.get_phonemes(word)
            if not phonemes:
                return word[-2:].lower()  # Fallback to last 2 characters
            
            # Find last stressed vowel and everything after it
            rhyme_start = -1
            for i in reversed(range(len(phonemes))):
                if any(char.isdigit() for char in phonemes[i]) and '1' in phonemes[i]:
                    rhyme_start = i
                    break
            
            if rhyme_start >= 0:
                rhyme_sound = ''.join(phonemes[rhyme_start:])
            else:
                # If no stressed vowel found, use last vowel sound
                rhyme_sound = ''.join(phonemes[-2:]) if len(phonemes) >= 2 else ''.join(phonemes)
            
            return rhyme_sound.lower()
        except:
            return word[-2:].lower()
    
    def calculate_rhyme_strength(self, word1: str, word2: str) -> float:
        """Calculate rhyme strength between two words"""
        try:
            sound1 = self.get_rhyme_sound(word1)
            sound2 = self.get_rhyme_sound(word2)
            
            if sound1 == sound2:
                return 1.0
            
            # Calculate phonetic similarity
            phonemes1 = self.get_phonemes(word1)
            phonemes2 = self.get_phonemes(word2)
            
            if not phonemes1 or not phonemes2:
                return 0.0
            
            # Compare ending sounds
            min_len = min(len(phonemes1), len(phonemes2))
            matches = 0
            
            for i in range(1, min_len + 1):
                if phonemes1[-i] == phonemes2[-i]:
                    matches += 1
                else:
                    break
            
            return matches / max(len(phonemes1), len(phonemes2))
        except:
            return 0.0

class WordDatabase:
    """Database of words organized by themes and properties"""
    
    def __init__(self):
        self.theme_words = self._initialize_theme_words()
        self.rhyme_groups = self._build_rhyme_groups()
        self.emotional_words = self._initialize_emotional_words()
        self.phonetic_analyzer = PhoneticAnalyzer()
    
    def _initialize_theme_words(self) -> Dict[LyricsTheme, List[str]]:
        """Initialize word database by theme"""
        return {
            LyricsTheme.LOVE: [
                "heart", "soul", "kiss", "embrace", "forever", "devotion", "passion",
                "tender", "gentle", "beautiful", "cherish", "adore", "beloved", "romance",
                "dreams", "stars", "moonlight", "sunrise", "together", "unity", "bond"
            ],
            LyricsTheme.HEARTBREAK: [
                "tears", "pain", "lonely", "empty", "broken", "shattered", "lost",
                "goodbye", "memories", "regret", "sorrow", "ache", "wounds", "cry",
                "darkness", "rain", "cold", "silence", "distance", "apart", "ending"
            ],
            LyricsTheme.FREEDOM: [
                "fly", "soar", "wings", "open", "wild", "free", "liberation", "escape",
                "boundless", "infinite", "horizon", "journey", "adventure", "explore",
                "wind", "mountain", "ocean", "sky", "road", "path", "choice"
            ],
            LyricsTheme.HOPE: [
                "light", "dawn", "tomorrow", "dream", "believe", "faith", "strength",
                "rise", "overcome", "triumph", "courage", "persevere", "endure",
                "sunshine", "rainbow", "spring", "bloom", "grow", "heal", "renewal"
            ],
            LyricsTheme.REBELLION: [
                "fight", "resist", "break", "chains", "revolution", "change", "power",
                "stand", "voice", "shout", "demand", "justice", "truth", "defiance",
                "storm", "fire", "thunder", "lightning", "war", "battle", "victory"
            ],
            LyricsTheme.PARTY: [
                "dance", "music", "loud", "celebrate", "fun", "night", "lights",
                "energy", "rhythm", "beat", "crowd", "friends", "laughter", "joy",
                "disco", "club", "party", "weekend", "wild", "crazy", "alive"
            ]
        }
    
    def _build_rhyme_groups(self) -> Dict[str, List[str]]:
        """Build groups of rhyming words"""
        rhyme_groups = defaultdict(list)
        
        # Sample rhyme groups
        common_rhymes = {
            "ay": ["day", "way", "say", "play", "stay", "may", "ray", "pray", "gray", "bay"],
            "ight": ["light", "night", "bright", "sight", "might", "fight", "right", "tight", "flight", "height"],
            "ove": ["love", "above", "dove", "shove", "glove"],
            "eart": ["heart", "start", "part", "art", "smart", "apart"],
            "ound": ["sound", "found", "ground", "round", "bound", "profound"],
            "ime": ["time", "rhyme", "climb", "prime", "sublime", "chime"],
            "ead": ["head", "said", "bread", "dead", "lead", "thread", "red"],
            "ame": ["name", "game", "fame", "shame", "blame", "flame", "same"]
        }
        
        for sound, words in common_rhymes.items():
            for word in words:
                rhyme_groups[sound].append(word)
        
        return dict(rhyme_groups)
    
    def _initialize_emotional_words(self) -> Dict[str, float]:
        """Initialize emotional weight mapping"""
        return {
            # Positive emotions
            "love": 0.9, "joy": 0.9, "happy": 0.8, "beautiful": 0.8, "wonderful": 0.8,
            "amazing": 0.8, "brilliant": 0.7, "good": 0.6, "nice": 0.5, "ok": 0.3,
            
            # Negative emotions
            "hate": -0.9, "pain": -0.8, "sad": -0.7, "angry": -0.7, "hurt": -0.8,
            "broken": -0.8, "lonely": -0.7, "empty": -0.6, "bad": -0.6, "wrong": -0.5,
            
            # Neutral words
            "and": 0.0, "the": 0.0, "is": 0.0, "in": 0.0, "to": 0.0, "of": 0.0
        }
    
    def get_theme_words(self, theme: LyricsTheme, count: int = 20) -> List[str]:
        """Get words related to a theme"""
        theme_words = self.theme_words.get(theme, [])
        if len(theme_words) <= count:
            return theme_words
        return random.sample(theme_words, count)
    
    def find_rhyming_words(self, word: str, max_count: int = 10) -> List[Tuple[str, float]]:
        """Find words that rhyme with the given word"""
        rhyme_sound = self.phonetic_analyzer.get_rhyme_sound(word)
        candidates = []
        
        # Check rhyme groups
        for sound, words in self.rhyme_groups.items():
            for candidate in words:
                if candidate != word.lower():
                    strength = self.phonetic_analyzer.calculate_rhyme_strength(word, candidate)
                    if strength > 0.5:
                        candidates.append((candidate, strength))
        
        # Sort by rhyme strength
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_count]
    
    def get_emotional_weight(self, word: str) -> float:
        """Get emotional weight of a word"""
        return self.emotional_words.get(word.lower(), 0.0)

class RhymeSchemeGenerator:
    """Generates and analyzes rhyme schemes"""
    
    def __init__(self):
        self.common_schemes = {
            RhymeScheme.AABB: ["A", "A", "B", "B"],
            RhymeScheme.ABAB: ["A", "B", "A", "B"],
            RhymeScheme.ABCB: ["A", "B", "C", "B"],
            RhymeScheme.AAAA: ["A", "A", "A", "A"],
            RhymeScheme.ABBA: ["A", "B", "B", "A"],
            RhymeScheme.AABA: ["A", "A", "B", "A"]
        }
    
    def generate_pattern(self, scheme: RhymeScheme, line_count: int = 4) -> RhymePattern:
        """Generate a rhyme pattern"""
        if scheme in self.common_schemes:
            base_pattern = self.common_schemes[scheme]
            # Extend or truncate pattern to match line count
            if line_count != len(base_pattern):
                pattern = (base_pattern * ((line_count // len(base_pattern)) + 1))[:line_count]
            else:
                pattern = base_pattern.copy()
        else:
            # Free verse - no rhyme scheme
            pattern = ["X"] * line_count
        
        # Generate syllable counts and stress patterns
        syllable_counts = [8, 8, 8, 8]  # Default to 8 syllables per line
        stress_patterns = ["iambic"] * line_count
        
        return RhymePattern(
            scheme=scheme,
            pattern=pattern,
            syllable_counts=syllable_counts,
            stress_patterns=stress_patterns
        )
    
    def analyze_existing_scheme(self, lines: List[str]) -> RhymeScheme:
        """Analyze the rhyme scheme of existing lines"""
        if len(lines) < 2:
            return RhymeScheme.FREE
        
        phonetic_analyzer = PhoneticAnalyzer()
        
        # Get rhyme sounds for each line
        rhyme_sounds = []
        for line in lines:
            words = line.strip().split()
            if words:
                last_word = words[-1].strip('.,!?;:')
                rhyme_sound = phonetic_analyzer.get_rhyme_sound(last_word)
                rhyme_sounds.append(rhyme_sound)
            else:
                rhyme_sounds.append("")
        
        # Determine pattern
        pattern = []
        sound_to_letter = {}
        current_letter = 'A'
        
        for sound in rhyme_sounds:
            if sound in sound_to_letter:
                pattern.append(sound_to_letter[sound])
            else:
                sound_to_letter[sound] = current_letter
                pattern.append(current_letter)
                current_letter = chr(ord(current_letter) + 1)
        
        # Match to known schemes
        pattern_str = ''.join(pattern).lower()
        
        for scheme, known_pattern in self.common_schemes.items():
            if pattern_str == ''.join(known_pattern).lower():
                return scheme
        
        return RhymeScheme.FREE

class NarrativeCoherenceEngine:
    """Ensures narrative coherence in lyrics"""
    
    def __init__(self):
        self.story_templates = self._initialize_story_templates()
        self.transition_phrases = self._initialize_transitions()
    
    def _initialize_story_templates(self) -> Dict[LyricsTheme, NarrativeStructure]:
        """Initialize narrative templates by theme"""
        return {
            LyricsTheme.LOVE: NarrativeStructure(
                exposition="Meeting someone special",
                rising_action="Growing feelings and connection",
                climax="Declaration of love or deep moment",
                falling_action="Challenges or deeper understanding",
                resolution="Commitment or eternal bond",
                emotional_arc=EmotionalArc.RISING,
                character_development={"protagonist": "vulnerable to confident"},
                plot_points=["first meeting", "first date", "conflict", "resolution", "future together"]
            ),
            LyricsTheme.HEARTBREAK: NarrativeStructure(
                exposition="Relationship in trouble",
                rising_action="Growing problems and distance",
                climax="The breakup moment",
                falling_action="Processing the loss",
                resolution="Acceptance and moving forward",
                emotional_arc=EmotionalArc.FALLING,
                character_development={"protagonist": "attached to independent"},
                plot_points=["happy times", "first problems", "breakup", "grief", "healing"]
            ),
            LyricsTheme.FREEDOM: NarrativeStructure(
                exposition="Feeling trapped or constrained",
                rising_action="Desire for change grows",
                climax="Breaking free moment",
                falling_action="Exploring newfound freedom",
                resolution="Finding peace in liberty",
                emotional_arc=EmotionalArc.RISING,
                character_development={"protagonist": "constrained to liberated"},
                plot_points=["constraint", "awakening", "escape", "exploration", "self-discovery"]
            )
        }
    
    def _initialize_transitions(self) -> Dict[str, List[str]]:
        """Initialize transition phrases"""
        return {
            "temporal": ["then", "now", "later", "before", "after", "suddenly", "meanwhile"],
            "causal": ["because", "so", "therefore", "since", "as a result", "that's why"],
            "contrast": ["but", "however", "yet", "still", "though", "on the other hand"],
            "continuation": ["and", "also", "furthermore", "moreover", "in addition"],
            "emotional": ["feeling", "sensing", "knowing", "realizing", "understanding"]
        }
    
    def create_narrative_structure(self, theme: LyricsTheme, emotional_arc: EmotionalArc) -> NarrativeStructure:
        """Create a narrative structure for given theme and arc"""
        if theme in self.story_templates:
            template = self.story_templates[theme]
            # Modify template based on emotional arc
            if emotional_arc != template.emotional_arc:
                template.emotional_arc = emotional_arc
                # Adjust plot points based on new arc
                template = self._adjust_for_emotional_arc(template, emotional_arc)
            return template
        else:
            # Create generic structure
            return NarrativeStructure(
                exposition="Setting the scene",
                rising_action="Building tension",
                climax="Peak moment",
                falling_action="Consequences",
                resolution="Conclusion",
                emotional_arc=emotional_arc,
                character_development={"protagonist": "changed"},
                plot_points=["beginning", "development", "peak", "resolution", "end"]
            )
    
    def _adjust_for_emotional_arc(self, structure: NarrativeStructure, arc: EmotionalArc) -> NarrativeStructure:
        """Adjust narrative structure for different emotional arcs"""
        if arc == EmotionalArc.U_SHAPED:
            # Start low, go high
            structure.exposition = "Starting from a low point"
            structure.resolution = "Reaching a high point"
        elif arc == EmotionalArc.INVERTED_U:
            # Start high, go low
            structure.exposition = "Starting from a high point"
            structure.resolution = "Reaching a low point"
        elif arc == EmotionalArc.STABLE:
            # Maintain consistent emotion
            structure.exposition = "Establishing consistent mood"
            structure.resolution = "Maintaining the mood"
        
        return structure
    
    def analyze_narrative_coherence(self, verses: List[LyricVerse]) -> LyricalCoherence:
        """Analyze the narrative coherence of lyrics"""
        if not verses:
            return LyricalCoherence(0, 0, 0, 0, 0, 0)
        
        # Thematic consistency
        themes = [verse.theme_focus for verse in verses]
        theme_counts = Counter(themes)
        dominant_theme_ratio = theme_counts.most_common(1)[0][1] / len(themes)
        thematic_consistency = dominant_theme_ratio
        
        # Narrative flow
        narrative_flow = self._calculate_narrative_flow(verses)
        
        # Emotional progression
        emotional_progression = self._calculate_emotional_progression(verses)
        
        # Linguistic coherence
        linguistic_coherence = self._calculate_linguistic_coherence(verses)
        
        # Structural integrity
        structural_integrity = self._calculate_structural_integrity(verses)
        
        # Overall score
        overall_score = np.mean([
            thematic_consistency,
            narrative_flow,
            emotional_progression,
            linguistic_coherence,
            structural_integrity
        ])
        
        return LyricalCoherence(
            thematic_consistency=thematic_consistency,
            narrative_flow=narrative_flow,
            emotional_progression=emotional_progression,
            linguistic_coherence=linguistic_coherence,
            structural_integrity=structural_integrity,
            overall_score=overall_score
        )
    
    def _calculate_narrative_flow(self, verses: List[LyricVerse]) -> float:
        """Calculate narrative flow score"""
        if len(verses) < 2:
            return 1.0
        
        flow_score = 0.0
        for i in range(len(verses) - 1):
            current_verse = verses[i]
            next_verse = verses[i + 1]
            
            # Check if narrative functions follow logical progression
            progression_score = self._check_narrative_progression(
                current_verse.narrative_function,
                next_verse.narrative_function
            )
            flow_score += progression_score
        
        return flow_score / (len(verses) - 1)
    
    def _calculate_emotional_progression(self, verses: List[LyricVerse]) -> float:
        """Calculate emotional progression score"""
        if len(verses) < 2:
            return 1.0
        
        intensities = [verse.emotional_intensity for verse in verses]
        
        # Check for appropriate emotional progression
        # Simple metric: variation should be meaningful but not chaotic
        intensity_variation = np.std(intensities)
        
        # Normalize variation score (0.2-0.4 is considered good variation)
        if 0.2 <= intensity_variation <= 0.4:
            return 1.0
        elif intensity_variation < 0.2:
            return 0.7  # Too stable
        else:
            return max(0.3, 1.0 - (intensity_variation - 0.4) * 2)  # Too chaotic
    
    def _calculate_linguistic_coherence(self, verses: List[LyricVerse]) -> float:
        """Calculate linguistic coherence score"""
        all_lines = []
        for verse in verses:
            all_lines.extend([line.text for line in verse.lines])
        
        if not all_lines:
            return 0.0
        
        # Simple linguistic coherence based on word repetition and theme consistency
        all_words = []
        for line in all_lines:
            words = re.findall(r'\w+', line.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        unique_words = len(word_counts)
        total_words = len(all_words)
        
        # Calculate repetition rate (moderate repetition is good for coherence)
        repetition_rate = 1 - (unique_words / total_words)
        
        # Optimal repetition rate is around 0.3-0.5
        if 0.3 <= repetition_rate <= 0.5:
            return 1.0
        else:
            return max(0.5, 1.0 - abs(repetition_rate - 0.4) * 2)
    
    def _calculate_structural_integrity(self, verses: List[LyricVerse]) -> float:
        """Calculate structural integrity score"""
        if not verses:
            return 0.0
        
        # Check for balanced structure
        verse_types = [verse.verse_type for verse in verses]
        type_counts = Counter(verse_types)
        
        # Good structure typically has verses and choruses
        has_verse = 'verse' in type_counts
        has_chorus = 'chorus' in type_counts
        
        structure_score = 0.5
        if has_verse and has_chorus:
            structure_score = 1.0
        elif has_verse or has_chorus:
            structure_score = 0.8
        
        # Check rhyme scheme consistency within verses
        rhyme_consistency = self._check_rhyme_consistency(verses)
        
        return (structure_score + rhyme_consistency) / 2
    
    def _check_narrative_progression(self, current_function: str, next_function: str) -> float:
        """Check if narrative functions progress logically"""
        progression_map = {
            "intro": ["exposition", "verse"],
            "exposition": ["rising_action", "verse"],
            "verse": ["chorus", "verse", "bridge"],
            "chorus": ["verse", "bridge", "outro"],
            "bridge": ["chorus", "outro"],
            "outro": ["end"]
        }
        
        if current_function in progression_map:
            if next_function in progression_map[current_function]:
                return 1.0
            else:
                return 0.5
        
        return 0.7  # Neutral for unknown functions
    
    def _check_rhyme_consistency(self, verses: List[LyricVerse]) -> float:
        """Check rhyme scheme consistency"""
        scheme_generator = RhymeSchemeGenerator()
        consistency_scores = []
        
        for verse in verses:
            if len(verse.lines) >= 2:
                line_texts = [line.text for line in verse.lines]
                detected_scheme = scheme_generator.analyze_existing_scheme(line_texts)
                
                # Check if detected scheme matches intended scheme
                if detected_scheme == verse.rhyme_scheme:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.6)  # Partial credit
        
        return np.mean(consistency_scores) if consistency_scores else 0.5

class AdvancedLyricsAI:
    """Main lyrics AI system combining all components"""
    
    def __init__(self):
        self.word_database = WordDatabase()
        self.rhyme_generator = RhymeSchemeGenerator()
        self.narrative_engine = NarrativeCoherenceEngine()
        self.phonetic_analyzer = PhoneticAnalyzer()
        logger.info("AdvancedLyricsAI initialized")
    
    def generate_lyrics(self, request: LyricsGenerationRequest) -> Dict[str, Any]:
        """Generate complete lyrics based on request"""
        try:
            logger.info(f"Generating lyrics for theme: {request.theme.value}")
            
            # Create narrative structure
            narrative = self.narrative_engine.create_narrative_structure(
                request.theme, request.emotional_arc
            )
            
            # Generate verses
            verses = []
            
            # Create verses based on request
            for i in range(request.verse_count):
                verse = self._generate_verse(request, narrative, i, "verse")
                verses.append(verse)
                
                # Add chorus after each verse (except last if bridge follows)
                if i < request.verse_count - 1 or request.bridge_count == 0:
                    chorus = self._generate_verse(request, narrative, i, "chorus")
                    verses.append(chorus)
            
            # Add bridge if requested
            if request.bridge_count > 0:
                bridge = self._generate_verse(request, narrative, 0, "bridge")
                verses.append(bridge)
                
                # Final chorus after bridge
                final_chorus = self._generate_verse(request, narrative, 0, "chorus")
                verses.append(final_chorus)
            
            # Analyze coherence
            coherence = self.narrative_engine.analyze_narrative_coherence(verses)
            
            # Format output
            lyrics_text = self._format_lyrics(verses)
            
            return {
                "lyrics": lyrics_text,
                "verses": [self._verse_to_dict(v) for v in verses],
                "narrative_structure": self._narrative_to_dict(narrative),
                "coherence_analysis": self._coherence_to_dict(coherence),
                "metadata": {
                    "theme": request.theme.value,
                    "style": request.style.value,
                    "rhyme_scheme": request.rhyme_scheme.value,
                    "emotional_arc": request.emotional_arc.value,
                    "word_count": len(lyrics_text.split()),
                    "line_count": sum(len(v.lines) for v in verses)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating lyrics: {e}")
            return self._get_default_lyrics(request)
    
    def _generate_verse(self, request: LyricsGenerationRequest, narrative: NarrativeStructure,
                       verse_index: int, verse_type: str) -> LyricVerse:
        """Generate a single verse"""
        try:
            # Determine emotional intensity based on arc and position
            emotional_intensity = self._calculate_emotional_intensity(
                request.emotional_arc, verse_index, verse_type
            )
            
            # Generate rhyme pattern
            rhyme_pattern = self.rhyme_generator.generate_pattern(request.rhyme_scheme, 4)
            
            # Generate lines
            lines = []
            rhyme_sounds = {}
            
            for i, pattern_letter in enumerate(rhyme_pattern.pattern):
                line = self._generate_line(
                    request, narrative, pattern_letter, rhyme_sounds,
                    emotional_intensity, verse_type, i
                )
                lines.append(line)
                
                # Store rhyme sound for pattern matching
                if pattern_letter != "X":  # X means no rhyme required
                    rhyme_sounds[pattern_letter] = line.rhyme_sound
            
            return LyricVerse(
                lines=lines,
                verse_type=verse_type,
                rhyme_scheme=request.rhyme_scheme,
                emotional_intensity=emotional_intensity,
                theme_focus=request.theme,
                narrative_function=self._get_narrative_function(verse_type, verse_index)
            )
            
        except Exception as e:
            logger.error(f"Error generating verse: {e}")
            return self._get_default_verse(request, verse_type)
    
    def _generate_line(self, request: LyricsGenerationRequest, narrative: NarrativeStructure,
                      pattern_letter: str, rhyme_sounds: Dict[str, str], 
                      emotional_intensity: float, verse_type: str, line_index: int) -> LyricLine:
        """Generate a single line"""
        try:
            # Get theme words
            theme_words = self.word_database.get_theme_words(request.theme, 50)
            
            # Determine target rhyme sound
            target_rhyme = rhyme_sounds.get(pattern_letter, "")
            
            # Generate line content
            if verse_type == "chorus":
                line_text = self._generate_chorus_line(theme_words, emotional_intensity, line_index)
            elif verse_type == "bridge":
                line_text = self._generate_bridge_line(theme_words, emotional_intensity, line_index)
            else:  # verse
                line_text = self._generate_verse_line(theme_words, emotional_intensity, line_index, narrative)
            
            # Adjust for rhyme if needed
            if target_rhyme and pattern_letter != "X":
                line_text = self._adjust_for_rhyme(line_text, target_rhyme, theme_words)
            
            # Extract properties
            words = line_text.split()
            last_word = words[-1] if words else "word"
            rhyme_sound = self.phonetic_analyzer.get_rhyme_sound(last_word)
            syllable_count = self._count_syllables(line_text)
            phonemes = self.phonetic_analyzer.get_phonemes(last_word)
            
            # Calculate emotional weight
            emotional_weight = np.mean([
                self.word_database.get_emotional_weight(word) 
                for word in words
            ]) if words else 0.0
            
            return LyricLine(
                text=line_text,
                rhyme_sound=rhyme_sound,
                syllable_count=syllable_count,
                stress_pattern="iambic",  # Simplified
                emotional_weight=emotional_weight,
                theme_relevance=0.8,  # Simplified
                phonemes=phonemes
            )
            
        except Exception as e:
            logger.error(f"Error generating line: {e}")
            return LyricLine(
                text="In the silence of the night",
                rhyme_sound="ight",
                syllable_count=8,
                stress_pattern="iambic",
                emotional_weight=0.2,
                theme_relevance=0.5,
                phonemes=["AY", "T"]
            )
    
    def _generate_verse_line(self, theme_words: List[str], emotional_intensity: float,
                           line_index: int, narrative: NarrativeStructure) -> str:
        """Generate a verse line"""
        # Simple template-based generation
        templates = [
            "In the {adj} {noun} of {time}",
            "When the {noun} starts to {verb}",
            "I can {verb} the {noun} {adj}",
            "Through the {noun} and the {noun2}",
            "Like a {noun} in the {place}",
            "Every {time} I {verb} about",
            "The {noun} that we used to {verb}",
            "Now I {verb} in {place}"
        ]
        
        template = random.choice(templates)
        
        # Fill template with theme-appropriate words
        replacements = {
            "adj": random.choice(["gentle", "bright", "dark", "sweet", "cold", "warm"]),
            "noun": random.choice(theme_words[:10]),
            "noun2": random.choice(theme_words[5:15]),
            "verb": random.choice(["feel", "see", "know", "find", "hold", "keep"]),
            "time": random.choice(["morning", "evening", "night", "day", "moment"]),
            "place": random.choice(["distance", "darkness", "light", "silence", "world"])
        }
        
        line = template
        for key, value in replacements.items():
            line = line.replace(f"{{{key}}}", value)
        
        return line
    
    def _generate_chorus_line(self, theme_words: List[str], emotional_intensity: float,
                            line_index: int) -> str:
        """Generate a chorus line"""
        # Chorus lines are typically more repetitive and impactful
        templates = [
            "We are {adj} tonight",
            "This is our {noun} to {verb}",
            "I will {verb} you {adv}",
            "Together we will {verb}",
            "Nothing can {verb} us now",
            "In this {noun} we {verb}",
            "Hold me {adj} and {adj2}",
            "Let the {noun} {verb} away"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            "adj": random.choice(["strong", "free", "alive", "brave", "true"]),
            "adj2": random.choice(["close", "tight", "near", "safe"]),
            "noun": random.choice(theme_words[:5]),
            "verb": random.choice(["dance", "sing", "fly", "run", "shine"]),
            "adv": random.choice(["forever", "always", "completely", "deeply"])
        }
        
        line = template
        for key, value in replacements.items():
            line = line.replace(f"{{{key}}}", value)
        
        return line
    
    def _generate_bridge_line(self, theme_words: List[str], emotional_intensity: float,
                            line_index: int) -> str:
        """Generate a bridge line"""
        # Bridge lines provide contrast and often contain the key message
        templates = [
            "But in the end we {verb}",
            "Maybe {noun} is all we {verb}",
            "Through it all I {verb}",
            "This is where we {verb}",
            "Beyond the {noun} and {noun2}",
            "When the {noun} fades away",
            "Here in this {adj} {noun}",
            "All that matters is the {noun}"
        ]
        
        template = random.choice(templates)
        
        replacements = {
            "adj": random.choice(["perfect", "sacred", "quiet", "magical"]),
            "noun": random.choice(theme_words[:8]),
            "noun2": random.choice(theme_words[3:11]),
            "verb": random.choice(["know", "understand", "believe", "find", "stand"])
        }
        
        line = template
        for key, value in replacements.items():
            line = line.replace(f"{{{key}}}", value)
        
        return line
    
    def _adjust_for_rhyme(self, line: str, target_rhyme: str, theme_words: List[str]) -> str:
        """Adjust line to match target rhyme"""
        try:
            # Find rhyming words
            rhyming_candidates = self.word_database.find_rhyming_words(target_rhyme)
            
            if rhyming_candidates:
                # Replace last word with rhyming word
                words = line.split()
                if words:
                    # Try to find a thematically appropriate rhyming word
                    best_rhyme = rhyming_candidates[0][0]  # Default to best rhyme
                    
                    for word, strength in rhyming_candidates:
                        if word in theme_words:
                            best_rhyme = word
                            break
                    
                    words[-1] = best_rhyme
                    return " ".join(words)
            
            return line
            
        except Exception as e:
            logger.error(f"Error adjusting for rhyme: {e}")
            return line
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified)"""
        try:
            words = re.findall(r'\w+', text.lower())
            total_syllables = 0
            
            for word in words:
                # Simple syllable counting
                vowels = 'aeiouy'
                syllables = 0
                prev_was_vowel = False
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        syllables += 1
                    prev_was_vowel = is_vowel
                
                # Handle silent e
                if word.endswith('e') and syllables > 1:
                    syllables -= 1
                
                # Ensure at least 1 syllable
                syllables = max(1, syllables)
                total_syllables += syllables
            
            return total_syllables
            
        except:
            return 8  # Default syllable count
    
    def _calculate_emotional_intensity(self, arc: EmotionalArc, verse_index: int, verse_type: str) -> float:
        """Calculate emotional intensity based on arc and position"""
        base_intensity = 0.5
        
        if verse_type == "chorus":
            base_intensity = 0.8  # Choruses are typically more intense
        elif verse_type == "bridge":
            base_intensity = 0.9  # Bridges are often the most intense
        
        # Adjust based on emotional arc
        if arc == EmotionalArc.RISING:
            intensity_modifier = 0.3 + (verse_index * 0.2)
        elif arc == EmotionalArc.FALLING:
            intensity_modifier = 0.9 - (verse_index * 0.2)
        elif arc == EmotionalArc.U_SHAPED:
            if verse_index < 2:
                intensity_modifier = 0.3
            else:
                intensity_modifier = 0.8
        elif arc == EmotionalArc.INVERTED_U:
            if verse_index < 2:
                intensity_modifier = 0.8
            else:
                intensity_modifier = 0.3
        else:  # STABLE or COMPLEX
            intensity_modifier = 0.6
        
        final_intensity = base_intensity * intensity_modifier
        return max(0.1, min(1.0, final_intensity))
    
    def _get_narrative_function(self, verse_type: str, verse_index: int) -> str:
        """Get narrative function for verse"""
        if verse_type == "verse":
            if verse_index == 0:
                return "exposition"
            elif verse_index == 1:
                return "rising_action"
            else:
                return "development"
        elif verse_type == "chorus":
            return "emotional_core"
        elif verse_type == "bridge":
            return "climax"
        else:
            return "transition"
    
    def _format_lyrics(self, verses: List[LyricVerse]) -> str:
        """Format verses into complete lyrics"""
        formatted_lines = []
        
        for verse in verses:
            # Add verse type label
            formatted_lines.append(f"[{verse.verse_type.title()}]")
            
            # Add lines
            for line in verse.lines:
                formatted_lines.append(line.text)
            
            # Add blank line between sections
            formatted_lines.append("")
        
        return "\n".join(formatted_lines).strip()
    
    def _verse_to_dict(self, verse: LyricVerse) -> Dict[str, Any]:
        """Convert verse to dictionary"""
        return {
            "verse_type": verse.verse_type,
            "rhyme_scheme": verse.rhyme_scheme.value,
            "emotional_intensity": verse.emotional_intensity,
            "theme_focus": verse.theme_focus.value,
            "narrative_function": verse.narrative_function,
            "lines": [
                {
                    "text": line.text,
                    "syllable_count": line.syllable_count,
                    "emotional_weight": line.emotional_weight,
                    "rhyme_sound": line.rhyme_sound
                }
                for line in verse.lines
            ]
        }
    
    def _narrative_to_dict(self, narrative: NarrativeStructure) -> Dict[str, Any]:
        """Convert narrative structure to dictionary"""
        return {
            "exposition": narrative.exposition,
            "rising_action": narrative.rising_action,
            "climax": narrative.climax,
            "falling_action": narrative.falling_action,
            "resolution": narrative.resolution,
            "emotional_arc": narrative.emotional_arc.value,
            "plot_points": narrative.plot_points
        }
    
    def _coherence_to_dict(self, coherence: LyricalCoherence) -> Dict[str, Any]:
        """Convert coherence analysis to dictionary"""
        return {
            "thematic_consistency": coherence.thematic_consistency,
            "narrative_flow": coherence.narrative_flow,
            "emotional_progression": coherence.emotional_progression,
            "linguistic_coherence": coherence.linguistic_coherence,
            "structural_integrity": coherence.structural_integrity,
            "overall_score": coherence.overall_score
        }
    
    def _get_default_lyrics(self, request: LyricsGenerationRequest) -> Dict[str, Any]:
        """Return default lyrics on error"""
        return {
            "lyrics": f"[Verse]\nIn the silence of the night\nI search for something bright\nThrough the shadows and the fear\nI know that love is near\n\n[Chorus]\nThis is our time to shine\nEverything will be fine\nTogether we will stand\nHand in hand",
            "verses": [],
            "narrative_structure": {},
            "coherence_analysis": {},
            "metadata": {
                "theme": request.theme.value,
                "style": request.style.value,
                "rhyme_scheme": request.rhyme_scheme.value,
                "word_count": 32,
                "line_count": 8
            }
        }
    
    def _get_default_verse(self, request: LyricsGenerationRequest, verse_type: str) -> LyricVerse:
        """Return default verse on error"""
        default_lines = [
            LyricLine("In the silence of the night", "ight", 8, "iambic", 0.2, 0.8, ["AY", "T"]),
            LyricLine("I search for something bright", "ight", 8, "iambic", 0.3, 0.8, ["AY", "T"]),
            LyricLine("Through the shadows and the fear", "ear", 8, "iambic", -0.2, 0.7, ["IH", "R"]),
            LyricLine("I know that love is near", "ear", 8, "iambic", 0.5, 0.9, ["IH", "R"])
        ]
        
        return LyricVerse(
            lines=default_lines,
            verse_type=verse_type,
            rhyme_scheme=request.rhyme_scheme,
            emotional_intensity=0.5,
            theme_focus=request.theme,
            narrative_function="development"
        )
    
    def analyze_existing_lyrics(self, lyrics_text: str) -> Dict[str, Any]:
        """Analyze existing lyrics for coherence and structure"""
        try:
            lines = [line.strip() for line in lyrics_text.split('\n') if line.strip()]
            
            # Simple verse detection (look for section markers)
            verses = []
            current_verse_lines = []
            current_verse_type = "verse"
            
            for line in lines:
                if line.startswith('[') and line.endswith(']'):
                    # Process previous verse
                    if current_verse_lines:
                        verse = self._create_verse_from_lines(current_verse_lines, current_verse_type)
                        verses.append(verse)
                    
                    # Start new verse
                    current_verse_type = line.strip('[]').lower()
                    current_verse_lines = []
                else:
                    current_verse_lines.append(line)
            
            # Process final verse
            if current_verse_lines:
                verse = self._create_verse_from_lines(current_verse_lines, current_verse_type)
                verses.append(verse)
            
            # Analyze coherence
            coherence = self.narrative_engine.analyze_narrative_coherence(verses)
            
            return {
                "coherence_analysis": self._coherence_to_dict(coherence),
                "structure_analysis": {
                    "verse_count": len([v for v in verses if v.verse_type == "verse"]),
                    "chorus_count": len([v for v in verses if v.verse_type == "chorus"]),
                    "bridge_count": len([v for v in verses if v.verse_type == "bridge"]),
                    "total_lines": sum(len(v.lines) for v in verses),
                    "average_syllables": np.mean([line.syllable_count for v in verses for line in v.lines])
                },
                "rhyme_analysis": {
                    "dominant_scheme": Counter([v.rhyme_scheme.value for v in verses]).most_common(1)[0][0] if verses else "unknown"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing lyrics: {e}")
            return {"error": str(e)}
    
    def _create_verse_from_lines(self, line_texts: List[str], verse_type: str) -> LyricVerse:
        """Create a verse from text lines"""
        lines = []
        
        for line_text in line_texts:
            words = line_text.split()
            last_word = words[-1] if words else "word"
            rhyme_sound = self.phonetic_analyzer.get_rhyme_sound(last_word)
            syllable_count = self._count_syllables(line_text)
            phonemes = self.phonetic_analyzer.get_phonemes(last_word)
            
            emotional_weight = np.mean([
                self.word_database.get_emotional_weight(word)
                for word in words
            ]) if words else 0.0
            
            line = LyricLine(
                text=line_text,
                rhyme_sound=rhyme_sound,
                syllable_count=syllable_count,
                stress_pattern="iambic",
                emotional_weight=emotional_weight,
                theme_relevance=0.7,
                phonemes=phonemes
            )
            lines.append(line)
        
        # Detect rhyme scheme
        rhyme_scheme = self.rhyme_generator.analyze_existing_scheme([line.text for line in lines])
        
        return LyricVerse(
            lines=lines,
            verse_type=verse_type,
            rhyme_scheme=rhyme_scheme,
            emotional_intensity=np.mean([line.emotional_weight for line in lines]) if lines else 0.5,
            theme_focus=LyricsTheme.LOVE,  # Default theme
            narrative_function="development"
        )

# Convenience functions
def create_lyrics_ai() -> AdvancedLyricsAI:
    """Create and return AdvancedLyricsAI instance"""
    return AdvancedLyricsAI()

def generate_song_lyrics(theme: str, style: str = "narrative", rhyme_scheme: str = "abab",
                        emotional_arc: str = "u_shaped") -> Dict[str, Any]:
    """Quick lyrics generation function"""
    try:
        lyrics_ai = create_lyrics_ai()
        
        request = LyricsGenerationRequest(
            theme=LyricsTheme(theme.lower()),
            style=LyricStyle(style.lower()),
            rhyme_scheme=RhymeScheme(rhyme_scheme.lower()),
            emotional_arc=EmotionalArc(emotional_arc.lower())
        )
        
        return lyrics_ai.generate_lyrics(request)
        
    except Exception as e:
        logger.error(f"Error in quick lyrics generation: {e}")
        return {"error": str(e)}

def analyze_lyrics(lyrics_text: str) -> Dict[str, Any]:
    """Quick lyrics analysis function"""
    try:
        lyrics_ai = create_lyrics_ai()
        return lyrics_ai.analyze_existing_lyrics(lyrics_text)
        
    except Exception as e:
        logger.error(f"Error in quick lyrics analysis: {e}")
        return {"error": str(e)}