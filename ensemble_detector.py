










import re
from urllib.parse import urlparse
from difflib import SequenceMatcher
from typing import Tuple

class PhishingRuleDetector:

    
    @staticmethod
    def levenshtein_ratio(s1: str, s2: str) -> float:

        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        matcher = SequenceMatcher(None, s1, s2)
        return matcher.ratio()
    
    @staticmethod
    def is_brand_spoofing(url_part: str, brand: str, threshold: float = 0.85) -> bool:












        substitutions = {
            'l': ['1', 'i', '|'],          # paypa1, paypai
            'o': ['0'],                     # amaz0n
            'e': ['3'],                     # appl3, verc3l
            'a': ['4', '@'],                # p4ypal
            's': ['5', '$'],                # p4ss
            'g': ['9'],                     # goo9le
            'b': ['8'],                     # fa8ook
            'i': ['1', '!', 'j'],           # m1crosoft
            't': ['+'],                     # microso+t
            'z': ['2'],                     # amaz2n
        }
        

        if brand in url_part:
            return True
        

        similarity = PhishingRuleDetector.levenshtein_ratio(url_part, brand)
        if similarity >= threshold:
            return True
        


        test_variants = [url_part]  # Start with original
        
        for original_char, replacements in substitutions.items():
            for replacement in replacements:
                if replacement in url_part:

                    variant = url_part.replace(replacement, original_char)
                    test_variants.append(variant)
        

        for variant in test_variants:
            variant_sim = PhishingRuleDetector.levenshtein_ratio(variant, brand)
            if variant_sim >= 0.80:  # Lower threshold for variant matching
                return True
        
        return False
    

    CRITICAL_PATTERNS = {
        '/wp-admin/': 'WordPress admin panel exposure',
        '/wp-admin': 'WordPress admin panel exposure',
        '/admin/': 'Admin panel exposure',
        '/admin': 'Admin panel exposure',
        '/dev/': 'Development folder exposure',
        '/dev': 'Development folder exposure',
        '/includes/': 'Server includes directory exposure',
        '/includes': 'Server includes directory exposure',
        'confirmation.php': 'Phishing form confirmation',
        'verification.php': 'Phishing form verification',
        '/phpMyAdmin': 'Database admin exposure',
    }
    

    SUSPICIOUS_PATTERNS = {
        '/confirm': 'Confirmation page (phishing indicator)',
        '/verify': 'Verification page (phishing indicator)',
        '/update': 'Update/upgrade page (phishing indicator)',
        '/validate': 'Validation page (phishing indicator)',
        '/security': 'Security page (may be fake)',
    }
    

    PROTECTED_BRANDS = [

        'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'twitter', 
        'netflix', 'dropbox', 'icloud', 'outlook', 'gmail', 'yahoo', 'at&t', 'verizon',
        'wells fargo', 'chase', 'bank of america', 'citibank', 'hsbc', 'barclays', 'irs',

        'vercel', 'netlify', 'heroku', 'github', 'gitlab', 'bitbucket', 'railway', 
        'render', 'replit', 'glitch', 'firebase', 'aws', 'azure', 'digitalocean', 
        'linode', 'cloudflare',

        'stripe', 'twilio', 'sendgrid', 'mailchimp', 'telegram', 'discord', 'whatsapp',
        'slack', 'linkedin'
    ]
    
    def __init__(self):

        self.critical_patterns = self.CRITICAL_PATTERNS
        self.suspicious_patterns = self.SUSPICIOUS_PATTERNS
        self.protected_brands = self.PROTECTED_BRANDS
    
    def detect(self, url: str) -> Tuple[str, float, str]:











        try:

            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            full_url = url.lower()
            

            domain_parts = domain.split('.')
            main_domain = domain_parts[0] if domain_parts else domain
            

            for pattern, reason in self.critical_patterns.items():
                if pattern.lower() in path:
                    return 'phishing', 1.0, f"CRITICAL: {reason}"
            

            for brand in self.protected_brands:

                if self.is_brand_spoofing(domain, brand, threshold=0.80) or \
                   self.is_brand_spoofing(main_domain, brand, threshold=0.80):
                    if brand not in domain.lower() and domain != brand:
                        return 'phishing', 0.95, f"Brand hijacking: {brand} spoofing detected"
            

            max_sus_conf = 0.0
            sus_reason = ""
            
            for pattern, reason in self.suspicious_patterns.items():
                if pattern.lower() in path:
                    sus_conf = 0.70  # Base confidence for suspicious patterns
                    if sus_conf > max_sus_conf:
                        max_sus_conf = sus_conf
                        sus_reason = reason
            
            if max_sus_conf > 0.0:
                return 'phishing', max_sus_conf, f"SUSPICIOUS: {sus_reason}"
            

            return 'legitimate', 0.95, "No obvious malicious patterns detected"
            
        except Exception as e:
            return 'legitimate', 0.95, f"Could not parse URL: {str(e)}"



def predict_url(url: str) -> Tuple[str, float, str]:

    detector = PhishingRuleDetector()
    return detector.detect(url)


if __name__ == "__main__":
    detector = PhishingRuleDetector()
    
    test_urls = [
        "https://github.com",
        "https://paypa1-security.com",
        "http://courgeon-immobilier.fr/dev/",
        "https://amazon.com",
    ]
    
    for url in test_urls:
        pred, conf, reason = detector.detect(url)
        print(f"{url}: {pred} ({conf:.0%}) - {reason}")
