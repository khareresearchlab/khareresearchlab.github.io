"""No-key RSS collector. Headline-only ranking; no AI-generated claims."""
import concurrent.futures
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OUTPUT = Path('assets/open-burning-news.json')
QUERIES = [
    '("stubble burning" OR "crop residue burning" OR "waste burning" OR "open burning") (pollution OR smoke OR emissions) India when:14d',
    '("open burning" OR "waste burning" OR "biomass burning") (pollution OR smoke OR emissions) when:14d',
    '("crop burning" OR "agricultural burning" OR "peat fires" OR "forest fires") ("air quality" OR pollution OR smoke) when:14d',
]
BURN = re.compile(r'open.burn|waste.burn|garbage.burn|trash.burn|stubble|crop.{0,20}burn|burn.{0,20}crop|biomass.burn|agricultural.burn|residue.burn|peat.fire|forest.fire|wildfire|burning', re.I)
AIR = re.compile(r'pollut|smog|smoke|emission|air.quality|pm2|toxic|haze', re.I)
INDIA = re.compile(r'\b(india|indian|delhi|punjab|haryana|kanpur|uttar pradesh|noida|gurugram|mumbai|bengaluru|bangalore|chennai|kolkata|hyderabad|pune|lucknow|ncr)\b', re.I)

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'KhareLabNews/1.0'}), timeout=20) as r:
        return r.read(3_000_000)

def parse_feed(raw, now):
    root = ET.fromstring(raw)
    if root.tag != 'rss' or root.find('channel') is None:
        raise ValueError('Not an RSS feed')
    results=[]
    for node in root.findall('./channel/item'):
        title = ' '.join((node.findtext('title') or '').split())
        source = ' '.join((node.findtext('source') or '').split())
        url = (node.findtext('link') or '').strip()
        if source and title.endswith(' - '+source): title=title[:-len(source)-3]
        try:
            dt=parsedate_to_datetime(node.findtext('pubDate') or '').astimezone(timezone.utc)
        except (ValueError,TypeError,OverflowError): continue
        if not (now-timedelta(days=14) <= dt <= now+timedelta(hours=1)): continue
        if not (BURN.search(title) and AIR.search(title)): continue
        if not source or urllib.parse.urlparse(url).scheme != 'https': continue
        results.append(dict(title=title,source=source,url=url,published_at=dt.isoformat(),region='India' if INDIA.search(title) else 'World'))
    return results

def select(items):
    ranked=sorted(items,key=lambda x:(x['region']=='India',x['published_at']),reverse=True)
    unique=[]
    for item in ranked:
        normalized=re.sub(r'\W+',' ',item['title'].lower()).strip()
        if any(item['url']==old['url'] or SequenceMatcher(None,normalized,re.sub(r'\W+',' ',old['title'].lower()).strip()).ratio()>.78 for old in unique): continue
        unique.append(item)
    india=[x for x in unique if x['region']=='India']
    world=[x for x in unique if x['region']!='India']
    chosen=india[:3]+world[:2]
    for item in unique:
        if len(chosen)>=5: break
        if item not in chosen: chosen.append(item)
    return sorted(chosen,key=lambda x:(x['region']=='India',x['published_at']),reverse=True)

def main():
    now=datetime.now(timezone.utc)
    # Restore the last deployed feed so a network failure does not erase it.
    prior=None
    try:
        prior=json.loads(get('https://khareresearchlab.github.io/assets/open-burning-news.json'))
    except Exception:
        if OUTPUT.exists():
            try: prior=json.loads(OUTPUT.read_text())
            except ValueError: pass
    urls=['https://news.google.com/rss/search?'+urllib.parse.urlencode({'q':q,'hl':'en-IN','gl':'IN','ceid':'IN:en'}) for q in QUERIES]
    items=[]; success=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(get,url) for url in urls]):
            try:
                items.extend(parse_feed(future.result(),now));success+=1
            except Exception as e: print('Feed unavailable:',type(e).__name__)
    if success < len(urls):
        if isinstance(prior,dict) and prior.get('updated_at') and isinstance(prior.get('items'),list):
            OUTPUT.parent.mkdir(parents=True,exist_ok=True)
            OUTPUT.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
            print('Partial/failed refresh: retained last successful feed.')
            return
        raise SystemExit('Could not complete initial refresh; deployment stopped. Retry workflow later.')
    result={'updated_at':now.isoformat(),'method':'RSS headline filtering; India prioritized','items':select(items)}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"Selected {len(result['items'])} recent headlines.")

if __name__=='__main__': main()
