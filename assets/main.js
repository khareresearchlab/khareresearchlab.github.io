const toggle=document.querySelector('.menu-toggle');const nav=document.querySelector('.nav');if(toggle&&nav){toggle.addEventListener('click',()=>{const open=nav.classList.toggle('is-open');toggle.setAttribute('aria-expanded',String(open));});}document.querySelectorAll('[data-year]').forEach(el=>el.textContent=new Date().getFullYear());

// Research news: all external text is inserted as text, never HTML.
document.querySelectorAll('.burning-news').forEach(async root => {
  const q = s => root.querySelector(s);
  const status = q('.burning-news-status');
  const date = value => new Date(value).toLocaleDateString('en-IN', {day:'numeric',month:'short',year:'numeric'});
  try {
    const response = await fetch(root.dataset.newsFeed || 'assets/open-burning-news.json', {cache:'no-cache'});
    if (!response.ok) throw new Error('Feed unavailable');
    const data = await response.json();
    const items = (Array.isArray(data.items) ? data.items : []).filter(item => {
      try { return typeof item.title === 'string' && typeof item.source === 'string' &&
        new URL(item.url).protocol === 'https:' && Number.isFinite(Date.parse(item.published_at)) &&
        Date.now()-Date.parse(item.published_at) < 14*86400000 && Date.parse(item.published_at) <= Date.now()+86400000;
      } catch { return false; }
    }).slice(0,5);
    const updated = Date.parse(data.updated_at);
    q('.burning-news-updated').textContent = Number.isFinite(updated) ?
      `Last checked ${date(updated)}${Date.now()-updated > 3*86400000 ? ' · Update delayed' : ''}` : '';
    if (!items.length) {
      status.textContent = data.updated_at ? 'No relevant coverage found in the past 14 days.' : 'News updates will appear after the first successful refresh.';
      return;
    }
    status.hidden = true;
    const link = q('.burning-news-link'); link.hidden = false;
    let index = 0, paused = false, hovering = false;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    paused = reduced.matches;
    const pauseButton = q('[data-news-pause]');
    const show = () => {
      const item = items[index];
      link.textContent = item.title; link.href = item.url;
      q('.burning-news-meta').textContent = `${item.region === 'India' ? 'INDIA' : 'WORLD'} · ${item.source} · ${date(item.published_at)}`;
      q('.burning-news-count').textContent = `${index+1} / ${items.length}`;
    };
    const move = n => {index = (index+n+items.length)%items.length; show();};
    const updateButton = () => {pauseButton.textContent = paused ? 'Play' : 'Pause'; pauseButton.setAttribute('aria-pressed',String(paused));};
    q('.burning-news-controls').hidden = items.length < 2;
    q('[data-news-prev]').addEventListener('click',()=>move(-1));
    q('[data-news-next]').addEventListener('click',()=>move(1));
    pauseButton.addEventListener('click',()=>{paused=!paused;updateButton();});
    root.addEventListener('mouseenter',()=>{hovering=true;});
    root.addEventListener('mouseleave',()=>{hovering=false;});
    reduced.addEventListener('change',e=>{paused=e.matches;updateButton();});
    show();updateButton();
    setInterval(()=>{if (!paused && !hovering && !document.hidden && !root.contains(document.activeElement)) move(1);},8000);
  } catch {
    status.textContent = 'News is temporarily unavailable. Please check again later.';
  }
});
