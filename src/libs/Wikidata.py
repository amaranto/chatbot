import sys
import logging
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)


CVE_WDT = "P3587"
MITRE_URL= "https://cveawg.mitre.org/api/cve/"

class Wikidata:
    def __init__(self,
                 endpoint_url: str = "https://query.wikidata.org/sparql",
                 user_agent: str = "WDQS Agent/%s.%s" % (sys.version_info[0], sys.version_info[1])
    ) -> None:
        self.endpoint_url = endpoint_url
        self.user_agent = user_agent
        self.sparql = SPARQLWrapper(endpoint_url)

    def get_item_id_from_url(self, query: str)-> str:
        return query.split("/")[-1]
    
    def get_results(self, query: str)-> dict:
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logger.error(e)
            return {}

    
    def get_property(self, value: str, p_id: str|None = None, lng: str = "en")->dict:
        
        p_id = CVE_WDT if not p_id else p_id

        query = f"""
        SELECT ?item ?itemLabel ?id ?itemDescription ?sitelink WHERE {{
            ?item wdt:{p_id} ?id.
            filter ( str(?id) = "{value}")
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lng}". }}
        }}
        """
        results = self.get_results(query)
        return results
    
    def get_item(self, item_id: str,lng: str = "en", limit=1)->dict:

        query = f"""
        select distinct ?property ?propertyLabel ?subject ?itemDescription ?itemLabel{{
            values (?item) {{(wd:{item_id})}}
            ?subject ?predicate ?item .
            ?property wikibase:directClaim ?predicate
            service wikibase:label {{ bd:serviceParam wikibase:language "{lng}" }}
        }} limit {limit}
        """
        results = self.get_results(query)
        return results

    def get_cve(self, value: str, uri: str = MITRE_URL)->str|None:
        try:
            r = requests.get(uri + value)
            if r.status_code == 200:
                response = r.json()

                containers = response["containers"]
                cna= containers["cna"]
                descriptions = cna["descriptions"][0]
                cve = descriptions["value"]
                logger.info(f"Mitre cve {cve}")

                return cve
            else:
                return None
            
        except Exception as e:
            logger.error(e)
            return None
        
