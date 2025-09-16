"""
Secure Suno AI Integration for Son1kVers3
Enhanced with cookie management, rate limiting, and fallback mechanisms
"""

import os
import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import secrets

logger = logging.getLogger(__name__)

@dataclass
class SunoCredentials:
    """Secure container for Suno credentials"""
    session_id: str
    cookie: str
    base_url: str = "https://studio-api.suno.ai"
    
    def __post_init__(self):
        # Mask sensitive data in logs
        self.masked_session = f"{self.session_id[:8]}...{self.session_id[-4:]}" if len(self.session_id) > 12 else "***"
        self.masked_cookie = f"{self.cookie[:8]}...{self.cookie[-4:]}" if len(self.cookie) > 12 else "***"

@dataclass
class SunoResponse:
    """Standardized Suno API response"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    job_id: Optional[str] = None
    audio_url: Optional[str] = None
    metadata: Optional[Dict] = None

class RateLimiter:
    """Advanced rate limiter for Suno API calls"""
    
    def __init__(self, calls_per_minute: int = 30, burst_limit: int = 5):
        self.calls_per_minute = calls_per_minute
        self.burst_limit = burst_limit
        self.calls = []
        self.burst_calls = []
        
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Clean old calls
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        self.burst_calls = [call_time for call_time in self.burst_calls if now - call_time < 10]
        
        # Check burst limit (5 calls per 10 seconds)
        if len(self.burst_calls) >= self.burst_limit:
            wait_time = 10 - (now - self.burst_calls[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for burst limit")
                await asyncio.sleep(wait_time)
                
        # Check minute limit
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for minute limit")
                await asyncio.sleep(wait_time)
                
        # Record this call
        now = time.time()
        self.calls.append(now)
        self.burst_calls.append(now)

class SecureSunoClient:
    """
    Enhanced Suno AI client with security, reliability and fallback mechanisms
    """
    
    def __init__(self, credentials: Optional[SunoCredentials] = None):
        self.credentials = credentials or self._load_credentials()
        self.rate_limiter = RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_available = False
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        
        # Security headers
        self.headers = {
            'User-Agent': 'Son1kVers3/3.0.0 (Music Generation Platform)',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        if self.credentials:
            self.headers.update({
                'Cookie': self.credentials.cookie,
                'Authorization': f'Bearer {self.credentials.session_id}'
            })
            
        logger.info(f"SunoClient initialized with credentials: {getattr(self.credentials, 'masked_session', 'None')}")
    
    @staticmethod
    def _load_credentials() -> Optional[SunoCredentials]:
        """Load credentials from environment variables"""
        session_id = os.getenv('SUNO_SESSION_ID')
        cookie = os.getenv('SUNO_COOKIE')
        base_url = os.getenv('SUNO_BASE_URL', 'https://studio-api.suno.ai')
        
        if not session_id or not cookie:
            logger.warning("Suno credentials not found in environment variables")
            return None
            
        return SunoCredentials(session_id=session_id, cookie=cookie, base_url=base_url)
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=int(os.getenv('SUNO_TIMEOUT', '120')))
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
        await self.health_check()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> bool:
        """Check if Suno API is available"""
        if not self.credentials:
            return False
            
        now = time.time()
        if now - self.last_health_check < self.health_check_interval and self.is_available:
            return True
            
        try:
            if not self.session:
                return False
                
            async with self.session.get(f"{self.credentials.base_url}/api/get_credits") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Suno health check OK - Credits: {data.get('credits_left', 'unknown')}")
                    self.is_available = True
                    self.last_health_check = now
                    return True
                else:
                    logger.warning(f"Suno health check failed: {response.status}")
                    self.is_available = False
                    return False
                    
        except Exception as e:
            logger.error(f"Suno health check error: {e}")
            self.is_available = False
            return False
    
    async def generate_music(
        self, 
        prompt: str, 
        duration: float = 8.0, 
        style: str = "auto",
        temperature: float = 1.0,
        **kwargs
    ) -> SunoResponse:
        """
        Generate music using Suno AI with comprehensive error handling
        """
        if not await self.health_check():
            return SunoResponse(
                success=False, 
                error="Suno API unavailable - falling back to local generation"
            )
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Enhanced prompt processing
            enhanced_prompt = self._enhance_prompt(prompt, style, temperature)
            
            # Generate request ID for tracking
            request_id = self._generate_request_id()
            
            payload = {
                "prompt": enhanced_prompt,
                "make_instrumental": False,
                "wait_audio": False,
                "model_version": "v3.5",
                "gpt_description_prompt": enhanced_prompt,
                "mv": "chirp-v3-5",
                "duration": min(duration, 30),  # Suno max duration
                "request_id": request_id
            }
            
            logger.info(f"Suno generate request {request_id}: {enhanced_prompt[:100]}...")
            
            async with self.session.post(
                f"{self.credentials.base_url}/api/generate/v2/",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    job_id = data.get('id') or data.get('batch_id')
                    
                    if job_id:
                        # Poll for completion
                        audio_url, metadata = await self._poll_for_completion(job_id)
                        
                        if audio_url:
                            return SunoResponse(
                                success=True,
                                data=data,
                                job_id=job_id,
                                audio_url=audio_url,
                                metadata={
                                    'prompt': enhanced_prompt,
                                    'duration': duration,
                                    'style': style,
                                    'provider': 'suno',
                                    'model': 'chirp-v3-5',
                                    'request_id': request_id,
                                    **metadata
                                }
                            )
                    
                    return SunoResponse(
                        success=False,
                        error=f"No job ID returned from Suno API",
                        data=data
                    )
                
                else:
                    error_text = await response.text()
                    logger.error(f"Suno API error {response.status}: {error_text}")
                    return SunoResponse(
                        success=False,
                        error=f"Suno API error: {response.status} - {error_text}"
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Suno API timeout")
            return SunoResponse(success=False, error="Suno API timeout")
            
        except Exception as e:
            logger.error(f"Suno generation error: {e}")
            return SunoResponse(success=False, error=f"Suno error: {str(e)}")
    
    async def _poll_for_completion(self, job_id: str, max_wait: int = 120) -> Tuple[Optional[str], Dict]:
        """Poll Suno API for job completion"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                async with self.session.get(f"{self.credentials.base_url}/api/get?ids={job_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            item = data[0]
                            status = item.get('status')
                            
                            if status == 'complete':
                                audio_url = item.get('audio_url')
                                if audio_url:
                                    metadata = {
                                        'title': item.get('title', ''),
                                        'duration': item.get('duration', 0),
                                        'model': item.get('model_name', 'chirp-v3-5'),
                                        'created_at': item.get('created_at', ''),
                                        'gpt_description_prompt': item.get('gpt_description_prompt', ''),
                                        'status': status
                                    }
                                    return audio_url, metadata
                            
                            elif status in ['error', 'failed']:
                                logger.error(f"Suno job {job_id} failed: {item.get('error_message', 'Unknown error')}")
                                return None, {'error': item.get('error_message', 'Generation failed')}
                            
                            # Still processing
                            logger.info(f"Suno job {job_id} status: {status}")
                        
                await asyncio.sleep(3)  # Wait 3 seconds before next poll
                
            except Exception as e:
                logger.error(f"Error polling Suno job {job_id}: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"Suno job {job_id} timed out after {max_wait}s")
        return None, {'error': 'Generation timeout'}
    
    def _enhance_prompt(self, prompt: str, style: str = "auto", temperature: float = 1.0) -> str:
        """Enhance prompt for better Suno results"""
        enhanced = prompt.strip()
        
        # Add style information if not already present
        if style != "auto" and style.lower() not in enhanced.lower():
            style_map = {
                'pop': 'pop music, catchy melody',
                'rock': 'rock music, electric guitars, drums',
                'electronic': 'electronic music, synthesizers, electronic beats',
                'latin': 'latin music, spanish guitar, latin percussion',
                'acoustic': 'acoustic music, acoustic guitar, organic instruments',
                'hiphop': 'hip hop, rap beats, urban sound'
            }
            
            if style in style_map:
                enhanced = f"{enhanced}, {style_map[style]}"
        
        # Adjust for temperature (creativity)
        if temperature > 1.2:
            enhanced += ", experimental, creative, unique arrangement"
        elif temperature < 0.8:
            enhanced += ", classic style, traditional arrangement"
        
        return enhanced[:500]  # Suno prompt limit
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracking"""
        timestamp = str(int(time.time() * 1000))
        random_part = secrets.token_hex(8)
        return f"s3_{timestamp}_{random_part}"
    
    async def get_credits(self) -> Dict[str, Any]:
        """Get remaining Suno credits"""
        if not await self.health_check():
            return {'error': 'Suno API unavailable'}
        
        try:
            async with self.session.get(f"{self.credentials.base_url}/api/get_credits") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {'error': f'HTTP {response.status}'}
        except Exception as e:
            logger.error(f"Error getting credits: {e}")
            return {'error': str(e)}
    
    async def get_generations(self, limit: int = 20) -> List[Dict]:
        """Get recent generations from Suno"""
        if not await self.health_check():
            return []
        
        try:
            async with self.session.get(f"{self.credentials.base_url}/api/get?limit={limit}") as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else []
                else:
                    logger.error(f"Error getting generations: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting generations: {e}")
            return []


# Singleton instance for the application
_suno_client: Optional[SecureSunoClient] = None

def get_suno_client() -> Optional[SecureSunoClient]:
    """Get the global Suno client instance"""
    global _suno_client
    if _suno_client is None and os.getenv('SUNO_ENABLE', 'true').lower() == 'true':
        _suno_client = SecureSunoClient()
    return _suno_client

async def test_suno_integration():
    """Test Suno integration"""
    client = get_suno_client()
    if not client:
        print("❌ Suno client not available - check environment variables")
        return False
    
    try:
        async with client:
            # Test health check
            health = await client.health_check()
            print(f"🏥 Health check: {'✅ OK' if health else '❌ Failed'}")
            
            if health:
                # Test credits
                credits = await client.get_credits()
                print(f"💳 Credits: {credits}")
                
                # Test generations list
                generations = await client.get_generations(limit=5)
                print(f"🎵 Recent generations: {len(generations)} found")
                
            return health
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_suno_integration())