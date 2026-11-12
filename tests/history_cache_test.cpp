#include "history_cache.hpp"
#include <cassert>
#include <iostream>
int main() {
 using namespace strata::history;
 struct Image { int64_t start; };
 std::vector<int32_t> image_ids(2048, 1);
 std::fill(image_ids.begin()+100,image_ids.begin()+300, 42);
 assert(images_discarded(image_ids, std::vector<Image>{{100}}, Plan{64,768,768},42));
 assert(!images_discarded(image_ids, std::vector<Image>{{100}}, Plan{128,768,768},42));
 assert(!images_discarded(image_ids, std::vector<Image>{{100}}, Plan{64,200,768},42));
 assert(!images_discarded(image_ids, std::vector<Image>{{800}}, Plan{64,768,768},42));
 assert(!images_discarded(image_ids, std::vector<Image>{{-1}}, Plan{64,768,768},42));
 std::vector<int32_t> old(2048), next;
 for (size_t i=0;i<old.size();++i) old[i]=int32_t(i+1000);
 constexpr int32_t turn=99;
 old[64]=turn;old[768]=turn;
 next.insert(next.end(),old.begin(),old.begin()+64);
 next.insert(next.end(),old.begin()+768,old.begin()+1536);
 next.push_back(20);
 auto p=match(old,next,turn);
 assert(p.prefix==64&&p.source==768&&p.length==768);
 auto unaligned=next;unaligned.insert(unaligned.begin()+4,123);
 assert(!match(old,unaligned,turn).length);
 {
   std::vector<int32_t> a(2200);for(size_t j=0;j<a.size();++j)a[j]=int32_t(j+1000);
   a[65]=a[769]=turn;a[66]=a[770]=7;a[67]=a[771]=8;
   std::vector<int32_t> b(a.begin(),a.begin()+65);b.insert(b.end(),a.begin()+769,a.begin()+1537);b.push_back(55);
   auto q=match(a,b,turn);assert(q.prefix==68&&q.source==772&&q.length==764);
 }
 {
   auto visual_old=old;
   std::fill(visual_old.begin()+1301,visual_old.begin()+1305,42);
   std::vector<int32_t> visual_next(visual_old.begin(),visual_old.begin()+64);
   visual_next.insert(visual_next.end(),visual_old.begin()+768,visual_old.begin()+1800);
   visual_next.push_back(20);
   auto q=match(visual_old,visual_next,turn,42);
   assert(q.prefix==64&&q.source==768&&q.length==532);
   assert(images_discarded(visual_old,std::vector<Image>{{1301}},q,42));
   assert(visual_next[q.prefix+q.length]!=42); // reread whole pooled block before image
   visual_old[20]=visual_next[20]=42;
   assert(!match(visual_old,visual_next,turn,42).length);
 }
 {
   std::vector<int32_t> old_map(6000),fresh(6000),out;
   for(size_t j=0;j<2000;j++)for(size_t k=0;k<3;k++)old_map[j*3+k]=fresh[j*3+k]=int32_t(j);
   fresh[600*3+1]=607;fresh[600*3+2]=611;
   assert(compose_positions(old_map,fresh,64,768,532,out));
   assert(out[64*3]==768&&out[595*3]==1299);
   assert(out[600*3]==1304&&out[600*3+1]==1311&&out[600*3+2]==1315);
   // A later image-bearing continuation preserves ALL committed positions.
   std::vector<int32_t> continued;
   assert(compose_positions(out,fresh,0,0,900,continued));
   assert(std::equal(continued.begin(),continued.begin()+2700,out.begin()));
   const auto saved=continued;
   assert(!compose_positions(out,fresh,0,0,2000,continued));assert(continued==saved);
   old_map[1299*3]=INT32_MAX;
   assert(!compose_positions(old_map,fresh,64,768,531,continued));assert(continued==saved);
 }
 {
   std::vector<int32_t> ids{10,11,42,42,42,42,12,13};
   std::vector<int32_t> visual{0,0,0,1,1,1,20,20,20,20,20,21,20,21,20,20,21,21,22,22,22,23,23,23,24,24,24};
   const std::vector<Image> owner{{2}};
   assert(position_map_valid(visual,ids,owner,32));
   assert(!position_map_valid(visual,ids,std::vector<Image>{},32));
   assert(!position_map_valid(visual,ids,std::vector<Image>{{-1}},32));
   assert(!position_map_valid(visual,ids,owner,21));
   auto malformed=visual;malformed[4*3+2]=21;
   assert(!position_map_valid(malformed,ids,owner,32));
   malformed=visual;malformed.back()=25;
   assert(!position_map_valid(malformed,ids,owner,32));
   malformed=visual;malformed.pop_back();
   assert(!position_map_valid(malformed,ids,owner,32));
 }
 auto wrong=next;wrong[0]=-1;assert(!match(old,wrong,turn).length);
 auto short_run=next;short_run.resize(100);assert(!match(old,short_run,turn).length);
 auto exact=old;exact.push_back(1);assert(!match(old,exact,turn).length);
 next.pop_back();p=match(old,next,turn);assert(p.length==764); // never sample stale logits
 assert(positions_valid({},100));assert(positions_valid({0,1,70,71},3));
 assert(!positions_valid({0,1,1,2},3));assert(!positions_valid({0,-1},1));
 assert(!positions_valid({0,1,4},2,4));assert(!positions_valid({0,1},2));
 for (size_t stride : {size_t(16),size_t(512)}) {
   std::vector<uint8_t> data(2048*stride);
   for (size_t j=0;j<data.size();++j)data[j]=uint8_t((j*13+j/stride)%251);
   auto before=data;
   assert(compact_rows(data.data(),data.size(),stride,64,768,768));
   assert(std::equal(data.begin(),data.begin()+64*stride,before.begin()));
   assert(std::equal(data.begin()+64*stride,data.begin()+832*stride,before.begin()+768*stride));
   auto moved=data;
   assert(!compact_rows(data.data(),data.size(),stride,64,2000,100));assert(data==moved);
   assert(!compact_rows(data.data(),data.size(),stride,768,64,768));assert(data==moved);
 }
 std::cout << "history matcher and position guards passed\n";
}
