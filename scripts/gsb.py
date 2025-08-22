import os
import sys
import json
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config


class GOOGLE_SAFE_BROWSING:
    def __init__(self, url):
        self.url = url
        self.endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={Config.GSB_API_KEY}"
        self.payload = {
            "client": {"clientId": "yourcompany", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        self.headers = {"Content-Type": "application/json"}
        
    def check_url(self):
        """
        Kiểm tra URL bằng Google Safe Browsing API
        Returns:
            dict: Kết quả kiểm tra với label và thông tin chi tiết
        """
        try:
            self.response = requests.post(
                self.endpoint, 
                headers=self.headers, 
                data=json.dumps(self.payload),
                timeout=10
            )
            
            if self.response.status_code == 200:
                result = self.response.json()
                return self._analyze_response(result)
            else:
                return {
                    "url": self.url,
                    "label": "unknown",
                    "confidence": 0.0,
                    "threats": [],
                    "status": "error",
                    "error": f"API request failed with status {self.response.status_code}",
                    "details": self.response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "url": self.url,
                "label": "unknown",
                "confidence": 0.0,
                "threats": [],
                "status": "error",
                "error": f"Request failed: {str(e)}",
                "details": None
            }
    
    def _analyze_response(self, response):
        """
        Phân tích response từ GSB API và trả về label phân loại
        Args:
            response: Response từ GSB API
        Returns:
            dict: Kết quả phân loại với label và confidence
        """
        # Nếu không có threats, URL là benign
        if not response or "matches" not in response:
            return {
                "url": self.url,
                "label": "benign",
                "confidence": 1.0,
                "threats": [],
                "status": "safe",
                "details": "No threats detected"
            }
        
        matches = response.get("matches", [])
        if not matches:
            return {
                "url": self.url,
                "label": "benign",
                "confidence": 1.0,
                "threats": [],
                "status": "safe",
                "details": "No threats detected"
            }
        
        # Phân tích các threats để xác định label
        threat_types = []
        for match in matches:
            threat_type = match.get("threatType", "")
            threat_types.append(threat_type)
        
        # Mapping threat types sang labels
        label, confidence = self._map_threats_to_label(threat_types)
        
        return {
            "url": self.url,
            "label": label,
            "confidence": confidence,
            "threats": threat_types,
            "status": "malicious",
            "details": f"Detected threats: {', '.join(threat_types)}"
        }
    
    def _map_threats_to_label(self, threat_types):
        """
        Map các threat types từ GSB sang labels của hệ thống
        Args:
            threat_types: List các threat types từ GSB
        Returns:
            tuple: (label, confidence)
        """
        # Mapping rules
        threat_to_label = {
            "MALWARE": "malware",
            "SOCIAL_ENGINEERING": "phishing", 
            "UNWANTED_SOFTWARE": "malware",
            "POTENTIALLY_HARMFUL_APPLICATION": "malware"
        }
        
        # Tìm label phù hợp nhất
        label_counts = {}
        for threat in threat_types:
            if threat in threat_to_label:
                label = threat_to_label[threat]
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if not label_counts:
            return "unknown", 0.5
        
        # Chọn label có số lượng cao nhất
        most_common_label = max(label_counts, key=label_counts.get)
        confidence = min(0.9, 0.5 + (label_counts[most_common_label] * 0.2))
        
        return most_common_label, confidence


def check_url_with_gsb(url):
    """
    Hàm tiện ích để kiểm tra URL bằng GSB
    Args:
        url (str): URL cần kiểm tra
    Returns:
        dict: Kết quả kiểm tra
    """
    gsb = GOOGLE_SAFE_BROWSING(url)
    return gsb.check_url()


if __name__ == "__main__":
    # Test với một URL mẫu
    test_url = "https://account.microsoft.com/account"
    print(f"Testing URL: {test_url}")
    
    result = check_url_with_gsb(test_url)
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    